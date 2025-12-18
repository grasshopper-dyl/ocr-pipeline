from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

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
app = FastAPI(title="docling-ocr-worker", version="1.2.1")


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

pipeline_options.table_structure_options.do_cell_matching = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
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
    perf_page_batch_size: int
    # convenience fields (so downstream doesn't need to parse confidence)
    mean_grade: Optional[str] = None
    low_grade: Optional[str] = None


class OcrResult(BaseModel):
    metadata: OcrMetadata
    confidence: Optional[ConfidenceReport] = None
    confidence_raw: Optional[Dict[str, Any]] = None
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
    h = hashlib.sha256()
    h.update(filename.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(data)
    return h.hexdigest()[:32]



def build_confidence_report(conv_result) -> Optional[ConfidenceReport]:
    conf = getattr(conv_result, "confidence", None)
    conf_dict = _to_dict(conf)
    if not conf_dict:
        return None

    pages_out: List[ConfidencePage] = []
    for p in (conf_dict.get("pages") or []):
        pages_out.append(
            ConfidencePage(
                table_grade=pd.get("table_grade"),
                mean_grade=pd.get("mean_grade"),
                low_grade=pd.get("low_grade"),
            )
        )

    return ConfidenceReport(
        table_grade=conf_dict.get("table_grade"),
        mean_grade=conf_dict.get("mean_grade"),
        low_grade=conf_dict.get("low_grade"),
        pages=pages_out,
    )


# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------


@app.post("/convert.ocr", response_model=OcrResult, response_model_exclude_none=True)
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


        pages = getattr(result, "pages", None) or getattr(doc, "pages", None) or []
        num_pages = len(pages)

        markdown = doc.export_to_markdown()

        confidence_obj = getattr(result, "confidence", None)
        confidence = build_confidence_report(result)

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
            mean_grade=(confidence.mean_grade if confidence else None),
            low_grade=(confidence.low_grade if confidence else None),
        )

        return OcrResult(
            metadata=meta,
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
