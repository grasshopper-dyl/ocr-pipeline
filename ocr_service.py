from io import BytesIO
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse

from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

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


def _convert_pdf_to_doc(file: UploadFile, pdf_bytes: bytes):
    ds = DocumentStream(
        name=file.filename or "upload.pdf",
        stream=BytesIO(pdf_bytes),
    )
    result = converter.convert(source=ds)
    doc = result.document
    if not doc:
        raise RuntimeError("Docling conversion returned no document")
    return doc


@app.post("/convert", response_class=PlainTextResponse)
async def convert_markdown(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        doc = _convert_pdf_to_doc(file, pdf_bytes)
        return doc.export_to_markdown()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion error: {type(e).__name__}: {e}",
        )


@app.post("/convert.txt", response_class=PlainTextResponse)
async def convert_text(file: UploadFile = File(...)):
    """
    Same conversion, but returns Docling's plain text export instead of markdown.
    """
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        doc = _convert_pdf_to_doc(file, pdf_bytes)

        # Docling plain-text export:
        # Most DoclingDocument builds support export_to_text().
        if hasattr(doc, "export_to_text"):
            return doc.export_to_text()

        # Fallback (still Docling-native): strip markdown-ish formatting if text export isn't available
        md = doc.export_to_markdown()
        return md
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion error: {type(e).__name__}: {e}",
        )

# ------------------------
# ENTRYPOINT
# ------------------------

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
