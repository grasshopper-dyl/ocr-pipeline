from __future__ import annotations

import hashlib
import logging
import os
import statistics
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption

try:
    import docling  # type: ignore
    DOCLING_VERSION = getattr(docling, "__version__", "unknown")
except Exception:
    DOCLING_VERSION = "unknown"


# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
log = logging.getLogger("ocr-worker")


# ------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------
app = FastAPI(title="docling-ocr-worker", version="1.2.0")


# ------------------------------------------------------------------
# PERF SETTINGS
# ------------------------------------------------------------------
PAGE_BATCH_SIZE = int(os.getenv("OCR_PAGE_BATCH_SIZE", "1"))
settings.perf.page_batch_size = PAGE_BATCH_SIZE
settings.debug.profile_pipeline_timings = True


# ------------------------------------------------------------------
# DOC LING: PDF + TESSERACT CLI OCR (ENGLISH) + TABLE STRUCTURE
# ------------------------------------------------------------------
OCR_LANG = (os.getenv("OCR_LANG", "eng") or "eng").strip()
PIPELINE_NAME = "docling_pdf_tesseract_cli_eng_tables"

ocr_options = TesseractCliOcrOptions(lang=[OCR_LANG])

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    force_full_page_ocr=True,
    ocr_options=ocr_options,
    do_table_structure=True,
)

# table tuning
pipeline_options.table_structure_options.do_cell_matching = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    }
)


# ------------------------------------------------------------------
# MODELS
# ------------------------------------------------------------------
class TimingStats(BaseModel):
    count: int
    min: float
    median: float
    max: float


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
    pages: List[ConfidencePage] = Field(default_factory=list)


class OcrMetadata(BaseModel):
    doc_id: str
    sha256: str
    filename: str
    content_type: Optional[str] = None
    byte_size: int
    received_at: str  # ISO8601
    num_pages: int
    status: str
    pipeline: str
    ocr_lang: List[str]
    table_structure: bool
    docling_version: str
    perf_page_batch_size: int


class OcrResult(BaseModel):
    metadata: OcrMetadata
    timings: Dict[str, TimingStats] = Field(default_factory=dict)
    confidence: Optional[ConfidenceReport] = None
    markdown: str


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def validate_pdf(file: UploadFile, data: bytes) -> None:
    fn = (file.filename or "").lower()
    if not fn.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_doc_id(filename: str, data: bytes) -> str:
    # stable short ID (same input => same doc_id)
    h = hashlib.sha256()
    h.update(filename.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(data)
    return h.hexdigest()[:32]


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if f != f:  # NaN
        return None
    if f in (float("inf"), float("-inf")):
        return None
    return f


def _to_dict(obj: Any) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):  # pydantic v2
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return None


def extract_timings(conv_result) -> Dict[str, TimingStats]:
    out: Dict[str, TimingStats] = {}
    timings = getattr(conv_result, "timings", None)
    if not isinstance(timings, dict):
        return out

    for key, item in timings.items():
        # Docling ProfilingItem typically has `times: List[float]`
        times = getattr(item, "times", None)
        if not isinstance(times, list) or not times:
            continue
        out[key] = TimingStats(
            count=len(times),
            min=float(min(times)),
            median=float(statistics.median(times)),
            max=float(max(times)),
        )
    return out


def build_confidence_report(conv_result) -> Optional[ConfidenceReport]:
    conf = getattr(conv_result, "confidence", None)
    conf_dict = _to_dict(conf)
    if not conf_dict:
        return None

    pages_out: List[ConfidencePage] = []
    for p in (conf_dict.get("pages") or []):
        pd = _to_dict(p) or {}
        pages_out.append(
            ConfidencePage(
                layout_score=_safe_float(pd.get("layout_score")),
                layout_grade=pd.get("layout_grade"),
                ocr_score=_safe_float(pd.get("ocr_score")),
                ocr_grade=pd.get("ocr_grade"),
                parse_score=_safe_float(pd.get("parse_score")),
                parse_grade=pd.get("parse_grade"),
                table_score=_safe_float(pd.get("table_score")),
                table_grade=pd.get("table_grade"),
                mean_grade=pd.get("mean_grade"),
                low_grade=pd.get("low_grade"),
            )
        )

    return ConfidenceReport(
        layout_score=_safe_float(conf_dict.get("layout_score")),
        layout_grade=conf_dict.get("layout_grade"),
        ocr_score=_safe_float(conf_dict.get("ocr_score")),
        ocr_grade=conf_dict.get("ocr_grade"),
        parse_score=_safe_float(conf_dict.get("parse_score")),
        parse_grade=conf_dict.get("parse_grade"),
        table_score=_safe_float(conf_dict.get("table_score")),
        table_grade=conf_dict.get("table_grade"),
        mean_grade=conf_dict.get("mean_grade"),
        low_grade=conf_dict.get("low_grade"),
        pages=pages_out,
    )


# Optional quality gate using Docling grades
GRADE_ORDER = {"POOR": 0, "FAIR": 1, "GOOD": 2, "EXCELLENT": 3}


def _grade_rank(g: Optional[str]) -> Optional[int]:
    if not g:
        return None
    return GRADE_ORDER.get(str(g).upper())


def enforce_min_grade(conf: Optional[ConfidenceReport]) -> None:
    """
    If OCR_MIN_GRADE is set (POOR/FAIR/GOOD/EXCELLENT), reject documents whose
    low_grade (preferred) or mean_grade falls below it. If Docling didn't
    provide grades, we do not block (you can tighten this later if needed).
    """
    min_grade = (os.getenv("OCR_MIN_GRADE") or "").strip().upper()
    if not min_grade:
        return

    target = _grade_rank(min_grade)
    if target is None:
        raise HTTPException(status_code=500, detail=f"Bad OCR_MIN_GRADE={min_grade}")

    if conf is None:
        return

    # Docling docs recommend focusing on mean_grade + low_grade
    observed = conf.low_grade or conf.mean_grade
    observed_rank = _grade_rank(observed)
    if observed_rank is None:
        return

    if observed_rank < target:
        raise HTTPException(
            status_code=422,
            detail=f"Conversion below threshold: observed={observed} required>={min_grade}",
        )


# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "pipeline": PIPELINE_NAME,
        "ocr_lang": [OCR_LANG],
        "table_structure": True,
        "confidence_scores": True,
        "docling_version": DOCLING_VERSION,
        "perf_page_batch_size": PAGE_BATCH_SIZE,
        "min_grade_gate": (os.getenv("OCR_MIN_GRADE") or "").strip().upper() or None,
    }


@app.post("/convert.ocr", response_model=OcrResult)
async def convert_ocr(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    validate_pdf(file, pdf_bytes)

    filename = file.filename or "upload.pdf"
    doc_id = make_doc_id(filename, pdf_bytes)
    digest = sha256_hex(pdf_bytes)

    received_at = datetime.now(timezone.utc).isoformat()

    stream = DocumentStream(name=filename, stream=BytesIO(pdf_bytes))

    try:
        result = converter.convert(source=stream)
        doc = getattr(result, "document", None)
        if not doc:
            raise RuntimeError("Docling returned no document")

        # Prefer Docling's own page list when available
        pages = getattr(result, "pages", None) or getattr(doc, "pages", None) or []
        num_pages = len(pages)

        markdown = doc.export_to_markdown()

        confidence = build_confidence_report(result)
        enforce_min_grade(confidence)

        meta = OcrMetadata(
            doc_id=doc_id,
            sha256=digest,
            filename=filename,
            content_type=getattr(file, "content_type", None),
            byte_size=len(pdf_bytes),
            received_at=received_at,
            num_pages=num_pages,
            status=str(getattr(result, "status", "UNKNOWN")),
            pipeline=PIPELINE_NAME,
            ocr_lang=[OCR_LANG],
            table_structure=True,
            docling_version=DOCLING_VERSION,
            perf_page_batch_size=PAGE_BATCH_SIZE,
        )

        return OcrResult(
            metadata=meta,
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

    log.info("Starting Docling OCR worker (Tesseract CLI, tables, confidence)")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
