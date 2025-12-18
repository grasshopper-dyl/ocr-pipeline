from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from io import BytesIO
from datetime import datetime, timezone
import hashlib
from typing import Any

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


app = FastAPI(title="docling-ocr-worker", version="0.3.1")


# --- Known working Docling config ---
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True
pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_doc_id(filename: str, data: bytes) -> str:
    h = hashlib.sha256()
    h.update(filename.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(data)
    return h.hexdigest()[:32]


def to_jsonable(x: Any) -> Any:
    """
    Docling sometimes returns numpy scalars / pydantic objects / custom types.
    This converts them into JSON-safe primitives.
    """
    return jsonable_encoder(x)


@app.post("/convert.ocr")
async def convert_ocr(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Empty upload")

    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    stream = DocumentStream(
        name=filename,
        stream=BytesIO(pdf_bytes),
    )

    try:
        result = converter.convert(source=stream)
        doc = result.document

        payload = {
            "doc_id": make_doc_id(filename, pdf_bytes),
            "filename": filename,
            "sha256": sha256_hex(pdf_bytes),
            "received_at": datetime.now(timezone.utc).isoformat(),
            "markdown": doc.export_to_markdown(),
            "confidence": to_jsonable(getattr(result, "confidence", None)),
            "status": to_jsonable(getattr(result, "status", None)),
        }

        # Important: return JSONResponse with encoded content
        return JSONResponse(content=to_jsonable(payload))

    except Exception as e:
        raise HTTPException(500, f"OCR failure: {type(e).__name__}: {e}")
