from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

app = FastAPI(title="ocr-service")

###### BASE CONFIG (UNCHANGED)
# - GraniteDocling model
# - Using the transformers framework

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
        ),
    }
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/convert", response_class=PlainTextResponse)
async def convert(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    # EXACT replacement for:
    # source = "https://arxiv.org/pdf/2501.17887"
    source = BytesIO(pdf_bytes)

    doc = converter.convert(source=source).document
    if not doc:
        raise HTTPException(status_code=500, detail="Conversion failed")

    return doc.export_to_markdown()
