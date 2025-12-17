from __future__ import annotations

import hashlib
import logging
import os
from io import BytesIO
from typing import Any, Dict, Optional, List

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
PAGE_BATCH_SIZE = int(os.getenv("OCR_PAGE_BATCH_SIZE", "1"))
settings.perf.page_batch_size = PAGE_BATCH_SIZE
settings.debug.profile_pipeline_timings = True


# ------------------------------------------------------------------
# DOC LING: PDF PIPELINE + TESSERACT CLI OCR (ENGLISH) + TABLES
# ------------------------------------------------------------------
ocr_options = TesseractCliOcrOptions(lang=["eng"])

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    force_full_page_ocr=True,
    ocr_options=ocr_options,
    do_table_structure=True,  # IMPORTANT for statements
)

# IMPORTANT for statements: improves table extraction quality
pipeline_options.table_structure_options.do_cell_matching = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)


# ------------------------------------------------------------------
# CONFIDENCE MODELS (DOC LING NATIVE REPORT SURFACE)
# ------------------------------------------------------------------
class ConfidencePage(BaseModel):
    # Component scores/grades (page-level)
    layout_score: Optional[float] = None
    layout_grade: Optional[str] = None

    ocr_score: Optional[float] = None
    ocr_grade: Optional[str] = None

    parse_score: Optional[float] = None
    parse_grade: Optional[str] = None

    table_score: Optional[float] = None
    table_grade: Optional[str] = None

    # Summary (page-level) — if present in your Docling version
    mean_grade: Optional[str] = None
    low_grade: Optional[str] = None


class ConfidenceReportOut(BaseModel):
    # Component scores/grades (document-level)
    layout_score: Optional[float] = None
    layout_grade: Optional[str] = None

    ocr_score: Optional[float] = None
    ocr_grade: Optional[str] = None

    parse_score: Optional[float] = None
    parse_grade: Optional[str] = None

    table_score: Optional[float] = None
    table_grade: Optional[str] = None

    # Summary grades (document-level)
    mean_grade: Optional[str] = None
    low_grade: Optional[str] = None

    # Page-level breakdown
    pages: List[ConfidencePage] = []


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
    confidence: Optional[ConfidenceReportOut] = None
    markdown: str


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
    """
    Keep timings useful for ops, but drop noisy legacy bits like doc_enrich.
    """
    out: Dict[str, Any] = {}

    if not (hasattr(conv_result, "timings") and isinstance(conv_result.timings, dict)):
        return out

    for k, v in conv_result.timings.items():
        # You asked to remove "enrich"
        if k == "doc_enrich":
            continue

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


def _to_dict_maybe(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Docling's confidence report is available at ConversionResult.confidence (v2.34.0+).
    It may be a Pydantic model or a dataclass-like object.
    """
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        # Pydantic v2 style
        return obj.model_dump()
    if hasattr(obj, "dict"):
        # Pydantic v1 style
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    # best-effort fallback
    try:
        return dict(obj)  # type: ignore[arg-type]
    except Exception:
        return None


def build_confidence_report(conv_result) -> Optional[ConfidenceReportOut]:
    """
    Normalize Docling's native confidence report into the fields you care about.
    """
    conf = getattr(conv_result, "confidence", None)
    conf_dict = _to_dict_maybe(conf)
    if not conf_dict:
        return None

    # Some versions nest pages under "pages" with same field names
    pages_out: List[ConfidencePage] = []
    pages = conf_dict.get("pages") or []
    if isinstance(pages, list):
        for p in pages:
            if not isinstance(p, dict):
                p = _to_dict_maybe(p) or {}
            pages_out.append(ConfidencePage(**(p or {})))

    # Root has document-level fields equally named (per docs)
    base = dict(conf_dict)
    base["pages"] = pages_out
    return ConfidenceReportOut(**base)


# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "pipeline": "docling_pdf_tesseract_cli_eng_tables",
        "page_batch_size": PAGE_BATCH_SIZE,
        "ocr_lang": ["eng"],
        "force_full_page_ocr": True,
        "do_table_structure": True,
        "do_cell_matching": True,
        "confidence_scores": True,  # requires docling >= 2.34.0
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
        if not (markdown or "").strip():
            raise HTTPException(status_code=502, detail="OCR produced empty output")

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

    log.info(
        "Starting Docling OCR (Tesseract CLI, eng, tables) on port %s",
        os.getenv("PORT", "8000"),
    )
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
