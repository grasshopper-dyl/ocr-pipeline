from __future__ import annotations

import hashlib
import re
from io import BytesIO
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


# ------------------------
# APP
# ------------------------

app = FastAPI(title="ocr-worker", version="0.1.0")


# ------------------------
# DEDICATED OCR (VLM) CONFIG
# ------------------------

# Keep conservative for stability; increase when your remote service is proven stable.
BATCH_SIZE = 16

settings.perf.page_batch_size = BATCH_SIZE
settings.debug.profile_pipeline_timings = True  # POC: include timings if available

vlm_opts = vlm_model_specs.GRANITEDOCLING_VLLM
vlm_opts.concurrency = BATCH_SIZE

pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_opts,
    enable_remote_services=True,  # required for remote inference service specs
)

converter_ocr = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }
)


# ------------------------
# RESPONSE MODEL
# ------------------------

class OcrResult(BaseModel):
    doc_id: str
    filename: str
    status: str
    num_pages: int
    pipeline: str
    timings: Dict[str, Any]
    metadata: Dict[str, Any]
    markdown: str
    text: str


# ------------------------
# HELPERS
# ------------------------

_MONEY_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
_ACCOUNT_RE = re.compile(r"\bAccount\s*#?\s*(\d{6,})\b|\bPrimary\s+Account\s+Number\b", re.IGNORECASE)


def _validate_pdf(file: UploadFile, pdf_bytes: bytes) -> None:
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")


def _make_doc_id(filename: str, pdf_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(filename.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(pdf_bytes)
    return h.hexdigest()[:32]


def _extract_metadata(text: str) -> Dict[str, Any]:
    """
    POC metadata extraction (lightweight, deterministic).
    You can replace this later with structured Docling doc traversal.
    """
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]
    money = _MONEY_RE.findall(text)
    dates = _DATE_RE.findall(text)

    # account number extraction: try to capture a number following "Account" if present
    acct_number: Optional[str] = None
    # common statement cue: look for a standalone long number line
    long_nums = [ln for ln in lines if re.fullmatch(r"\d{6,}", ln)]
    if long_nums:
        acct_number = long_nums[0]

    # if not found, attempt a more direct regex scan
    if acct_number is None:
        m = re.search(r"\bAccount\s*#?\s*(\d{6,})\b", text, flags=re.IGNORECASE)
        if m:
            acct_number = m.group(1)

    return {
        "has_image_marker": "<!-- image -->" in text,
        "account_number": acct_number,
        "dates": list(dict.fromkeys(dates))[:10],     # unique, cap
        "money_values": money[:20],                   # cap
        "line_count": len(lines),
    }


def _safe_timings(conv_result) -> Dict[str, Any]:
    """
    Docling timings can be rich objects. For POC, return the keys and whatever is JSON-friendly.
    If not available, return {}.
    """
    timings: Dict[str, Any] = {}

    if hasattr(conv_result, "timings") and isinstance(conv_result.timings, dict):
        # best-effort summary
        for k, v in conv_result.timings.items():
            # some ProfilingItem objects have .times
            if hasattr(v, "times"):
                times = getattr(v, "times", None)
                # times could be list[float]
                if isinstance(times, list):
                    timings[k] = {
                        "count": len(times),
                        "min": min(times) if times else None,
                        "median": sorted(times)[len(times) // 2] if times else None,
                        "max": max(times) if times else None,
                    }
                else:
                    timings[k] = str(v)
            else:
                timings[k] = str(v)

    return timings


# ------------------------
# ROUTES
# ------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/convert.ocr", response_model=OcrResult)
async def convert_ocr(file: UploadFile = File(...)):
    """
    Dedicated OCR endpoint (VLM pipeline).
    Returns Docling exports + metadata + conversion timings (POC).
    """
    pdf_bytes = await file.read()
    _validate_pdf(file, pdf_bytes)

    filename = file.filename or "upload.pdf"
    doc_id = _make_doc_id(filename, pdf_bytes)

    ds = DocumentStream(
        name=filename,
        stream=BytesIO(pdf_bytes),
    )

    try:
        conv_result = converter_ocr.convert(source=ds)
        doc = conv_result.document
        if not doc:
            raise RuntimeError("Docling conversion returned no document")

        markdown = doc.export_to_markdown()
        text = doc.export_to_text() if hasattr(doc, "export_to_text") else markdown

        num_pages = len(getattr(conv_result, "pages", []) or [])
        status = str(getattr(conv_result, "status", "UNKNOWN"))

        metadata = _extract_metadata(text)
        timings = _safe_timings(conv_result)

        return OcrResult(
            doc_id=doc_id,
            filename=filename,
            status=status,
            num_pages=num_pages,
            pipeline="vlm_ocr",
            timings=timings,
            metadata=metadata,
            markdown=markdown,
            text=text,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR conversion error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
