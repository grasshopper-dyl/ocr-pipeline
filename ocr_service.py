##stable version 1.0

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, ConfigDict, TypeAdapter

from io import BytesIO
from datetime import datetime, timezone
import hashlib
from typing import Any, Optional

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline


app = FastAPI(title="docling-ocr-worker", version="0.4.0")


class OcrResponse(BaseModel):
    # makes Pydantic accept non-primitive types and coerce when we dump
    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_id: str
    filename: str
    sha256: str
    received_at: str

    markdown: str

    # keep these optional + JSON-safe
    confidence: Optional[Any] = None
    status: Optional[Any] = None

    # optional debugging metadata
    accelerator: Optional[str] = None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_doc_id(filename: str, data: bytes) -> str:
    h = hashlib.sha256()
    h.update(filename.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(data)
    return h.hexdigest()[:32]


def dump_jsonable(x: Any) -> Any:
    """
    Uses Pydantic v2 TypeAdapter to safely convert:
    - enums
    - numpy scalars
    - pydantic models
    - dataclasses (if present)
    into plain JSON types.
    """
    return TypeAdapter(Any).dump_python(x, mode="json")


# --- Explicit CUDA + threaded pipeline (this is "where the accelerator is") ---
pipeline_options = ThreadedPdfPipelineOptions(
    accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
    ocr_batch_size=4,
    layout_batch_size=64,
    table_batch_size=4,
)
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True
pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=ThreadedStandardPdfPipeline,
            pipeline_options=pipeline_options,
        )
    }
)


@app.post("/convert.ocr", response_model=OcrResponse)
async def convert_ocr(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Empty upload")

    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    stream = DocumentStream(name=filename, stream=BytesIO(pdf_bytes))

    try:
        result = converter.convert(source=stream)
        doc = result.document

        resp = OcrResponse(
            doc_id=make_doc_id(filename, pdf_bytes),
            filename=filename,
            sha256=sha256_hex(pdf_bytes),
            received_at=datetime.now(timezone.utc).isoformat(),
            markdown=doc.export_to_markdown(),
            confidence=dump_jsonable(getattr(result, "confidence", None)),
            status=dump_jsonable(getattr(result, "status", None)),
            accelerator="cuda"  # you can set this dynamically if you want
        )

        # return model_dump(mode="json") to guarantee JSON primitives
        return resp.model_dump(mode="json")

    except Exception as e:
        raise HTTPException(500, f"OCR failure: {type(e).__name__}: {e}")
