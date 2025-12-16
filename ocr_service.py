from __future__ import annotations

import re
from io import BytesIO
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

import pdfplumber

from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


# ------------------------
# APP + CONVERTER
# ------------------------

app = FastAPI(title="ocr-service", version="1.0")

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
        ),
    }
)


# ------------------------
# HELPERS
# ------------------------

def _ensure_pdf(upload: UploadFile, pdf_bytes: bytes) -> None:
    name = (upload.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")


def _docling_convert_to_markdown(filename: str, pdf_bytes: bytes) -> str:
    ds = DocumentStream(
        name=filename or "upload.pdf",
        stream=BytesIO(pdf_bytes),
    )
    result = converter.convert(source=ds)
    doc = result.document
    if not doc:
        raise RuntimeError("Docling conversion returned no document")
    return doc.export_to_markdown()


def _pdf_text_layer(pdf_bytes: bytes) -> str:
    """
    Extracts the PDF text layer (no OCR). This is your ground-truth baseline
    for PDFs that already contain text (like statement_sample1.pdf).
    """
    out_lines: List[str] = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                out_lines.append(txt.strip())
    return "\n\n".join(out_lines).strip()


def _clean_markdown(md: str) -> str:
    """
    Conservative cleanup:
    - Normalize whitespace
    - Remove obvious repeated consecutive lines
    - Cap pathological repeats (e.g., "Total Checks Paid" spam)
    """
    # Normalize line endings
    md = md.replace("\r\n", "\n").replace("\r", "\n")

    # Unescape common HTML entities seen in your output
    md = md.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")

    lines = [ln.strip() for ln in md.split("\n")]

    cleaned: List[str] = []
    last = None

    # First pass: remove consecutive duplicates
    for ln in lines:
        if not ln:
            # keep single blank lines, but avoid many in a row
            if cleaned and cleaned[-1] == "":
                continue
            cleaned.append("")
            last = ""
            continue

        if ln == last:
            continue

        cleaned.append(ln)
        last = ln

    # Second pass: cap pathological repeats anywhere in doc
    # Example you hit: "Total Checks Paid" repeated hundreds of times
    capped: List[str] = []
    counts: Dict[str, int] = {}

    for ln in cleaned:
        key = ln

        # Treat "Total Checks Paid" lines as same bucket even if extra spaces
        if re.match(r"^Total\s+Checks\s+Paid\b", ln, flags=re.IGNORECASE):
            key = "TOTAL_CHECKS_PAID_LINE"

        counts[key] = counts.get(key, 0) + 1

        # Allow normal lines to repeat a bit (headers etc.), but not explode
        limit = 3
        if key == "TOTAL_CHECKS_PAID_LINE":
            limit = 1

        if counts[key] <= limit:
            capped.append(ln)

    # Final normalization: strip leading/trailing blank lines
    while capped and capped[0] == "":
        capped.pop(0)
    while capped and capped[-1] == "":
        capped.pop()

    return "\n".join(capped).strip()


# ------------------------
# RESPONSE MODELS
# ------------------------

class ConvertResponse(BaseModel):
    filename: str
    markdown_raw: str
    markdown_clean: str
    text_ground_truth: str


# ------------------------
# ROUTES
# ------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.post("/convert.md", response_class=PlainTextResponse)
async def convert_markdown(file: UploadFile = File(...)) -> str:
    """
    Returns cleaned markdown (plain text response) for quick use.
    """
    pdf_bytes = await file.read()
    _ensure_pdf(file, pdf_bytes)

    try:
        md = _docling_convert_to_markdown(file.filename or "upload.pdf", pdf_bytes)
        return _clean_markdown(md)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion error: {type(e).__name__}: {e}")


@app.post("/convert", response_model=ConvertResponse)
async def convert(file: UploadFile = File(...)) -> ConvertResponse:
    """
    Returns:
      - markdown_raw: direct Docling export
      - markdown_clean: deduped/normalized markdown (prevents spam lines)
      - text_ground_truth: PDF text layer extraction (baseline)
    """
    pdf_bytes = await file.read()
    _ensure_pdf(file, pdf_bytes)

    try:
        md_raw = _docling_convert_to_markdown(file.filename or "upload.pdf", pdf_bytes)
        md_clean = _clean_markdown(md_raw)
        txt_gt = _pdf_text_layer(pdf_bytes)

        return ConvertResponse(
            filename=file.filename or "upload.pdf",
            markdown_raw=md_raw,
            markdown_clean=md_clean,
            text_ground_truth=txt_gt,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
