from io import BytesIO
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse

from docling.datamodel.base_models import DocumentStream  # <-- add this import

@app.post("/convert", response_class=PlainTextResponse)
async def convert(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    # IMPORTANT: Docling wants Path/str/DocumentStream — not raw BytesIO
    ds = DocumentStream(
        name=file.filename or "upload.pdf",
        stream=BytesIO(pdf_bytes),
    )

    try:
        result = converter.convert(source=ds)
        doc = result.document
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion error: {type(e).__name__}: {e}")

    return doc.export_to_markdown()
