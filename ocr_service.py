from __future__ import annotations

import hashlib
import logging
import os
from io import BytesIO
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


# ------------------------
# LOGGING
# ------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
_log = logging.getLogger("ocr-worker")


# ------------------------
# APP
# ------------------------
app = FastAPI(title="ocr-worker", version="0.2.0")


# ------------------------
# PERF TUNING (LOCAL)
# ------------------------
PAGE_BATCH_SIZE = int(os.getenv("OCR_PAGE_BATCH_SIZE", "4"))
settings.perf.page_batch_size = PAGE_BATCH_SIZE
settings.debug.profile_pipeline_timings = True


# ------------------------
# LOCAL VLM CONVERTER
# ------------------------
# This uses Docling's local Transformers-based Granite-Docling VLM default
# (no ApiVlmOptions, no enable_remote_services).
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
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
    markdown: str
    text: str


# ------------------------
# HELPERS
# ------------------------
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


def _safe_timings(conv_result) -> Dict[str, Any]:
    timings: Dict[str, Any] = {}
    if hasattr(conv_result, "timings") and isinstance(conv_result.timings, dict):
        for k, v in conv_result.timings.items():
            if hasattr(v, "times"):
                times = getattr(v, "times", None)
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
    return {
        "ok": True,
        "pipeline": "docling_granite_258_local_transformers",
        "page_batch_size": PAGE_BATCH_SIZE,
        "note": "Local VlmPipeline (no remote VLM API).",
    }


@app.post("/convert.ocr", response_model=OcrResult)
async def convert_ocr(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    _validate_pdf(file, pdf_bytes)

    filename = file.filename or "upload.pdf"
    doc_id = _make_doc_id(filename, pdf_bytes)

    ds = DocumentStream(name=filename, stream=BytesIO(pdf_bytes))

    try:
        conv_result = converter.convert(source=ds)
        doc = conv_result.document
        if not doc:
            raise RuntimeError("Docling conversion returned no document")

        markdown = doc.export_to_markdown()
        text = doc.export_to_text() if hasattr(doc, "export_to_text") else markdown

        # Hard-fail on empty output (prevents silent SUCCESS + blank doc)
        if not (text or "").strip():
            raise HTTPException(
                status_code=502,
                detail="Empty OCR output from local VLM pipeline.",
            )

        num_pages = len(getattr(conv_result, "pages", []) or [])
        status = str(getattr(conv_result, "status", "UNKNOWN"))

        return OcrResult(
            doc_id=doc_id,
            filename=filename,
            status=status,
            num_pages=num_pages,
            pipeline="docling_granite_258_local_transformers",
            timings=_safe_timings(conv_result),
            markdown=markdown,
            text=text,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR conversion error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
