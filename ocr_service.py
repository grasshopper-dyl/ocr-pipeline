from __future__ import annotations

import hashlib
import logging
import os
from io import BytesIO
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption


# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
log = logging.getLogger("ocr-worker")


# ------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------
app = FastAPI(title="docling-ocr-worker", version="1.0.0")


# ------------------------------------------------------------------
# PERF SETTINGS
# ------------------------------------------------------------------
PAGE_BATCH_SIZE = int(os.getenv("OCR_PAGE_BATCH_SIZE", "1"))  # single-page statements
settings.perf.page_batch_size = PAGE_BATCH_SIZE
settings.debug.profile_pipeline_timings = True


# ------------------------------------------------------------------
# DOC LING: PDF PIPELINE + TESSERACT CLI OCR (ENGLISH) — SINGLE PAGE
# ------------------------------------------------------------------
# Your setup, but forced to English:
# - "eng" requires tesseract language data for English (usually installed by default).
ocr_options = TesseractCliOcrOptions(lang=["eng"])

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    force_full_page_ocr=True,
    ocr_options=ocr_options,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
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
                t = getattr(v, "times", None)
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
        "pipeline": "docling_pdf_tesseract_cli_eng",
        "page_batch_size": PAGE_BATCH_SIZE,
        "ocr_lang": ["eng"],
        "force_full_page_ocr": True,
    }


@app.post("/convert.ocr", response_model=OcrResult)
async def convert_ocr(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    validate_pdf(file, pdf_bytes)

    filename = file.filename or "upload.pdf"
    doc_id = make_doc_id(filename, pdf_bytes)

    stream = DocumentStream(name=filename, stream=BytesIO(pdf_bytes))

    try:
        result = converter.convert(source=stream)
        doc = result.document
        if not doc:
            raise RuntimeError("Docling returned no document")

        markdown = doc.export_to_markdown()
        text = doc.export_to_text() if hasattr(doc, "export_to_text") else markdown

        if not (text or "").strip():
            raise HTTPException(status_code=502, detail="OCR produced empty output")

        return OcrResult(
            doc_id=doc_id,
            filename=filename,
            status=str(getattr(result, "status", "UNKNOWN")),
            num_pages=len(getattr(result, "pages", []) or []),
            pipeline="docling_pdf_tesseract_cli_eng",
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

    log.info("Starting Docling OCR (Tesseract CLI, eng) on port %s", os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
