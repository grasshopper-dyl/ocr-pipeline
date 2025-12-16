from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse

from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
import uvicorn

# ------------------------
# REQUIRED GLOBALS
# ------------------------

app = FastAPI(title="ocr-service")

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
        ),
    }
)

# ------------------------
# ROUTES
# ------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/convert", response_class=PlainTextResponse)
async def convert(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    ds = DocumentStream(
        name=file.filename or "upload.pdf",
        stream=BytesIO(pdf_bytes),
    )

    try:
        result = converter.convert(source=ds)
        doc = result.document
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion error: {type(e).__name__}: {e}",
        )

    return doc.export_to_markdown()

# ------------------------
# ENTRYPOINT
# ------------------------

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
