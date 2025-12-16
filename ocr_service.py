from io import BytesIO
import re
import html

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
# OUTPUT CLEANUP (NO NEW DEPS)
# ------------------------

_LOC_TAIL_RE = re.compile(r"\bloc_\d+>\s*$", re.IGNORECASE)

def _clean_output(text: str) -> str:
    # 1) HTML entities -> real characters
    text = html.unescape(text)

    # 2) remove trailing loc_###>
    text = _LOC_TAIL_RE.sub("", text).rstrip()

    # 3) collapse repeated consecutive lines + cap pathological repeats
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    out = []
    last = None
    total_checks_paid_seen = 0

    for ln in lines:
        if not ln:
            # keep at most one blank line in a row
            if out and out[-1] == "":
                continue
            out.append("")
            last = ""
            continue

        # drop consecutive duplicates
        if ln == last:
            continue

        # cap pathological repeats anywhere (your exact failure case)
        if ln.lower() == "total checks paid":
            total_checks_paid_seen += 1
            if total_checks_paid_seen > 1:
                continue

        out.append(ln)
        last = ln

    # strip leading/trailing blank lines
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()

    return "\n".join(out)

# ------------------------
# ROUTES
# ------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/convert", response_class=PlainTextResponse)
async def convert_markdown(file: UploadFile = File(...)):
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

    # keep /convert as markdown (but cleaned)
    return _clean_output(doc.export_to_markdown())


@app.post("/convert.txt", response_class=PlainTextResponse)
async def convert_text(file: UploadFile = File(...)):
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

    # text export if available; otherwise fall back to markdown, still cleaned
    if hasattr(doc, "export_to_text"):
        return _clean_output(doc.export_to_text())

    return _clean_output(doc.export_to_markdown())

# ------------------------
# ENTRYPOINT
# ------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
