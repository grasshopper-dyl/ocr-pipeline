from __future__ import annotations

import hashlib
import logging
import os
import re
import html
from io import BytesIO
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
log = logging.getLogger("ocr-worker")


# ------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------
app = FastAPI(title="docling-ocr-worker", version="1.0.1")


# ------------------------------------------------------------------
# PERFORMANCE / PIPELINE SETTINGS (LOCAL ONLY)
# ------------------------------------------------------------------
PAGE_BATCH_SIZE = int(os.getenv("OCR_PAGE_BATCH_SIZE", "4"))
settings.perf.page_batch_size = PAGE_BATCH_SIZE
settings.debug.profile_pipeline_timings = True


# ------------------------------------------------------------------
# DOC LING LOCAL GRANITE VLM
# ------------------------------------------------------------------
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
        )
    }
)


# ------------------------------------------------------------------
# RESPONSE MODEL
# ------------------------------------------------------------------
class OcrResult(BaseModel):
    doc_id: str
    filename: str
    status: str
    num_pages: int
    pipeline: str
    timings: Dict[str, Any]
    markdown: str
    text: str


# ------------------------------------------------------------------
# CLEANUP / NORMALIZATION
# ------------------------------------------------------------------
# Match:
# - loc_499>
# - loc_499&gt;
# - loc\_499>
# - loc\_499&gt;
_LOC_TAG_RE = re.compile(r"\bloc\\?_?\d+(?:&gt;|>)", re.IGNORECASE)

# Also remove the literal substring "loc_###" even if formatting is odd
_LOC_BARE_RE = re.compile(r"\bloc\\?_?\d+\b", re.IGNORECASE)


def clean_output(s: str) -> str:
    """
    Deterministic cleanup for layout-heavy docs like bank statements:
    - HTML unescape
    - remove loc artifacts
    - normalize newlines
    - collapse excessive repetition:
        * consecutive duplicate cap
        * global per-line cap (keeps 'Total Checks Paid' from exploding)
    """
    if not s:
        return ""

    # Decode HTML entities (&amp; → &, &gt; → >, etc.)
    s = html.unescape(s)

    # Remove loc artifacts
    s = _LOC_TAG_RE.sub("", s)
    s = _LOC_BARE_RE.sub("", s)

    # Normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Split and trim
    raw_lines = [ln.strip() for ln in s.split("\n")]

    cleaned: list[str] = []
    prev: str | None = None
    consecutive_run = 0

    # Caps
    MAX_CONSEC_REPEAT = int(os.getenv("OCR_MAX_REPEAT_SAME_LINE", "2"))   # consecutive
    MAX_TOTAL_REPEAT = int(os.getenv("OCR_MAX_TOTAL_REPEAT_LINE", "3"))  # global per-line
    total_counts: Dict[str, int] = {}

    for ln in raw_lines:
        if not ln:
            # keep single blank lines only
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            prev = ""
            consecutive_run = 0
            continue

        # Global cap per identical line (prevents runaway loops)
        c = total_counts.get(ln, 0)
        if c >= MAX_TOTAL_REPEAT:
            continue
        total_counts[ln] = c + 1

        # Consecutive cap (extra safety)
        if ln == prev:
            consecutive_run += 1
            if consecutive_run >= MAX_CONSEC_REPEAT:
                continue
        else:
            prev = ln
            consecutive_run = 0

        cleaned.append(ln)

    # Trim trailing blank lines
    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned).strip()


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def validate_pdf(file: UploadFile, data: bytes) -> None:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")


def make_doc_id(filename: str, data: bytes) -> str:
    h = hashlib.sha256()
    h.update(filename.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(data)
    return h.hexdigest()[:32]


def extract_timings(conv_result) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if hasattr(conv_result, "timings") and isinstance(conv_result.timings, dict):
        for k, v in conv_result.timings.items():
            if hasattr(v, "times"):
                t = v.times
                if isinstance(t, list) and t:
                    out[k] = {
                        "count": len(t),
                        "min": min(t),
                        "median": sorted(t)[len(t) // 2],
                        "max": max(t),
                    }
                else:
                    out[k] = str(v)
            else:
                out[k] = str(v)
    return out


# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "pipeline": "docling_granite_258_local",
        "page_batch_size": PAGE_BATCH_SIZE,
        "backend": "transformers",
        "max_consecutive_repeat": int(os.getenv("OCR_MAX_REPEAT_SAME_LINE", "2")),
        "max_total_repeat_line": int(os.getenv("OCR_MAX_TOTAL_REPEAT_LINE", "3")),
    }


@app.post("/convert.ocr", response_model=OcrResult)
async def convert_ocr(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    validate_pdf(file, pdf_bytes)

    filename = file.filename or "upload.pdf"
    doc_id = make_doc_id(filename, pdf_bytes)

    stream = DocumentStream(
        name=filename,
        stream=BytesIO(pdf_bytes),
    )

    try:
        result = converter.convert(source=stream)
        doc = result.document
        if not doc:
            raise RuntimeError("Docling returned no document")

        raw_md = doc.export_to_markdown()
        raw_text = doc.export_to_text() if hasattr(doc, "export_to_text") else raw_md

        markdown = clean_output(raw_md)
        text = clean_output(raw_text)

        if not text:
            raise HTTPException(status_code=502, detail="OCR produced empty output after cleanup")

        return OcrResult(
            doc_id=doc_id,
            filename=filename,
            status=str(getattr(result, "status", "UNKNOWN")),
            num_pages=len(getattr(result, "pages", []) or []),
            pipeline="docling_granite_258_local",
            timings=extract_timings(result),
            markdown=markdown,
            text=text,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failure: {type(e).__name__}: {e}")


# ------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    log.info("Starting Docling OCR (Granite-258 local) on port %s", os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
