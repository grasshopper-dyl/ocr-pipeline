from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse

from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline


# ------------------------
# REQUIRED GLOBALS
# ------------------------

app = FastAPI(title="ocr-service")

# Threaded pipeline options (based on your sample)
pipeline_options = ThreadedPdfPipelineOptions(
    accelerator_options=AcceleratorOptions(
        device=AcceleratorDevice.CUDA,
    ),
    ocr_batch_size=4,
    layout_batch_size=64,
    table_batch_size=4,
)

# Key: prefer the embedded PDF text layer (no OCR)
pipeline_options.do_ocr = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=ThreadedStandardPdfPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)


# ------------------------
# HELPERS
# ------------------------

def _validate_pdf(file: UploadFile, pdf_bytes: bytes) -> None:
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")


def _convert_to_doc(file: UploadFile, pdf_bytes: bytes):
    ds = DocumentStream(
        name=file.filename or "upload.pdf",
        stream=BytesIO(pdf_bytes),
    )
    result = converter.convert(source=ds)
    doc = result.document
    if not doc:
        raise RuntimeError("Docling conversion returned no document")
    return doc


# ------------------------
# ROUTES
# ------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/convert", response_class=PlainTextResponse)
async def convert_markdown(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    _validate_pdf(file, pdf_bytes)

    try:
        doc = _convert_to_doc(file, pdf_bytes)
        return doc.export_to_markdown()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion error: {type(e).__name__}: {e}",
        )


@app.post("/convert.txt", response_class=PlainTextResponse)
async def convert_text(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    _validate_pdf(file, pdf_bytes)

    try:
        doc = _convert_to_doc(file, pdf_bytes)

        # If your Docling build supports this, you'll get a cleaner text dump.
        if hasattr(doc, "export_to_text"):
            return doc.export_to_text()

        # Fallback: still Docling-native
        return doc.export_to_markdown()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion error: {type(e).__name__}: {e}",
        )


# ------------------------
# ENTRYPOINT
# ------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
