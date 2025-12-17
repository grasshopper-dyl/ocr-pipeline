from __future__ import annotations

import hashlib
import logging
import math
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

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
app = FastAPI(title="docling-ocr-worker", version="1.1.0")


# ------------------------------------------------------------------
# PERF SETTINGS (single / low-page documents)
# ------------------------------------------------------------------
PAGE_BATCH_SIZE = int(os.getenv("OCR_PAGE_BATCH_SIZE", "1"))
settings.perf.page_batch_size = PAGE_BATCH_SIZE
settings.debug.profile_pipeline_timings = True


# ------------------------------------------------------------------
# DOC LING: PDF + TESSERACT CLI OCR (ENGLISH) + TABLE STRUCTURE
# ------------------------------------------------------------------
ocr_options = TesseractCliOcrOptions(lang=["eng"])

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    force_full_page_ocr=True,
    ocr_options=ocr_options,
    do_table_structure=True,
)

pipeline_options.table_structure_options.do_cell_matching = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)


# ------------------------------------------------------------------
# MODELS
# ------------------------------------------------------------------
class ConfidencePage(BaseModel):
    layout_score: Optional[float] = None
    layout_grade: Optional[str] = None
    ocr_score: Optional[float] = None
    ocr_grade: Optional[str] = None
    parse_score: Optional[float] = None
    parse_grade: Optional[str] = None
    table_score: Optional[float] = None
    table_grade: Optional[str] = None
    mean_grade: Optional[str] = None
    low_grade: Optional[str] = None


class ConfidenceReport(BaseModel):
    layout_score: Optional[float] = None
    layout_grade: Optional[str] = None
    ocr_score: Optional[float] = None
    ocr_grade: Optional[str] = None
    parse_score: Optional[float] = None
    parse_grade: Optional[str] = None
    table_score: Optional[float] = None
    table_grade: Optional[str] = None
    mean_grade: Optional[str] = None
    low_grade: Optional[str] = None
    pages: List[ConfidencePage] = []


class OcrResult(BaseModel):
    doc_id: str
    filename: str
    status: str
    num_pages: int
    pipeline: str
    timings: Dict[str, Any]
    confidence: Optional[ConfidenceReport]
    markdown: str


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def safe_float(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def to_dict(obj) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return None


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
    timings = getattr(conv_result, "timings", None)
    if isinstance(timings, dict):
        for k, v in timings.items():
            t = getattr(v, "times", None)
            if isinstance(t, list) and t:
                out[k] = {
                    "count": len(t),
                    "min": min(t),
                    "median": sorted(t)[len(t) // 2],
                    "max": max(t),
                }
    return out


def build_confidence_report(conv_result) -> Optional[ConfidenceReport]:
    conf = getattr(conv_result, "confidence", None)
    conf_dict = to_dict(conf)
    if not conf_dict:
        return None

    pages_out: List[ConfidencePage] = []
    for p in conf_dict.get("pages", []) or []:
        p = to_dict(p) or {}
        pages_out.append(
            ConfidencePage(
                layout_score=safe_float(p.get("layout_score")),
                layout_grade=p.get("layout_grade"),
                ocr_score=safe_float(p.get("ocr_score")),
                ocr_grade=p.get("ocr_grade"),
                parse_score=safe_float(p.get("parse_score")),
                parse_grade=p.get("parse_grade"),
                table_score=safe_float(p.get("table_score")),
                table_grade=p.get("table_grade"),
                mean_grade=p.get("mean_grade"),
                low_grade=p.get("low_grade"),
            )
        )

    return ConfidenceReport(
        layout_score=safe_float(conf_dict.get("layout_score")),
        layout_grade=conf_dict.get("layout_grade"),
        ocr_score=safe_float(conf_dict.get("ocr_score")),
        ocr_grade=conf_dict.get("ocr_grade"),
        parse_score=safe_float(conf_dict.get("parse_score")),
        parse_grade=conf_dict.get("parse_grade"),
        table_score=safe_float(conf_dict.get("table_score")),
        table_grade=conf_dict.get("table_grade"),
        mean_grade=conf_dict.get("mean_grade"),
        low_grade=conf_dict.get("low_grade"),
        pages=pages_out,
    )


# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "pipeline": "docling_pdf_tesseract_cli_eng_tables",
        "ocr_lang": ["eng"],
        "table_structure": True,
        "confidence_scores": True,
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

        confidence = build_confidence_report(result)

        return OcrResult(
            doc_id=doc_id,
            filename=filename,
            status=str(getattr(result, "status", "UNKNOWN")),
            num_pages=len(getattr(result, "pages", []) or []),
            pipeline="docling_pdf_tesseract_cli_eng_tables",
            timings=extract_timings(result),
            confidence=confidence,
            markdown=markdown,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR failure: {type(e).__name__}: {e}",
        )


# ------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    log.info("Starting Docling OCR (Tesseract CLI, eng, tables, confidence)")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
