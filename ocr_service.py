# ocr_service.py
from __future__ import annotations

import hashlib
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# ------------------------
# LOGGING
# ------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
_log = logging.getLogger("ocr-worker")

# ------------------------
# APP
# ------------------------
app = FastAPI(title="ocr-worker", version="0.1.0")

# ------------------------
# DOCling GRANITE 258 (REMOTE, OPENAI-COMPATIBLE)
# ------------------------
# vLLM default: localhost:8000
# LM Studio default: localhost:1234
OCR_HOSTPORT = os.getenv("OCR_HOSTPORT", "localhost:8000")

# For vLLM use: ibm-granite/granite-docling-258M
# For LM Studio MLX model name might be: granite-docling-258m-mlx
OCR_MODEL = os.getenv("OCR_MODEL", "ibm-granite/granite-docling-258M")

OCR_PROMPT = os.getenv("OCR_PROMPT", "Convert this page to docling.")
OCR_FORMAT = ResponseFormat.DOCTAGS

OCR_MAX_TOKENS = int(os.getenv("OCR_MAX_TOKENS", "4096"))
OCR_TEMPERATURE = float(os.getenv("OCR_TEMPERATURE", "0.2"))
OCR_API_KEY = os.getenv("OCR_API_KEY", "")
OCR_SKIP_SPECIAL_TOKENS = os.getenv("OCR_SKIP_SPECIAL_TOKENS", "true").lower() == "true"

PAGE_BATCH_SIZE = int(os.getenv("OCR_PAGE_BATCH_SIZE", "16"))
settings.perf.page_batch_size = PAGE_BATCH_SIZE
settings.debug.profile_pipeline_timings = True


def openai_compatible_vlm_options(
    model: str,
    prompt: str,
    format: ResponseFormat,
    hostname_and_port: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    api_key: str = "",
    skip_special_tokens: bool = True,
) -> ApiVlmOptions:
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return ApiVlmOptions(
        url=f"http://{hostname_and_port}/v1/chat/completions",
        params=dict(
            model=model,
            max_tokens=max_tokens,
            skip_special_tokens=skip_special_tokens,  # commonly needed for vLLM
        ),
        headers=headers,
        prompt=prompt,
        timeout=90,
        scale=2.0,
        temperature=temperature,
        response_format=format,
    )


pipeline_options = VlmPipelineOptions(enable_remote_services=True)
pipeline_options.vlm_options = openai_compatible_vlm_options(
    model=OCR_MODEL,
    hostname_and_port=OCR_HOSTPORT,
    prompt=OCR_PROMPT,
    format=OCR_FORMAT,
    api_key=OCR_API_KEY,
    temperature=OCR_TEMPERATURE,
    max_tokens=OCR_MAX_TOKENS,
    skip_special_tokens=OCR_SKIP_SPECIAL_TOKENS,
)

converter_ocr = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            pipeline_cls=VlmPipeline,
        )
    }
)

# ------------------------
# RESPONSE MODEL
# ------------------------
class OcrResult(BaseModel):
    doc_id: str
    filename: str
    status: str
    num_pages: int
    pipeline: str
    timings: Dict[str, Any]
    metadata: Dict[str, Any]
    markdown: str
    text: str


# ------------------------
# HELPERS
# ------------------------
_MONEY_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)


def _validate_pdf(file: UploadFile, pdf_bytes: bytes) -> None:
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")


def _make_doc_id(filename: str, pdf_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(filename.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(pdf_bytes)
    return h.hexdigest()[:32]


def _extract_metadata(text: str) -> Dict[str, Any]:
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]
    money = _MONEY_RE.findall(text)
    dates = _DATE_RE.findall(text)

    acct_number: Optional[str] = None
    long_nums = [ln for ln in lines if re.fullmatch(r"\d{6,}", ln)]
    if long_nums:
        acct_number = long_nums[0]
    if acct_number is None:
        m = re.search(r"\bAccount\s*#?\s*(\d{6,})\b", text, flags=re.IGNORECASE)
        if m:
            acct_number = m.group(1)

    return {
        "account_number": acct_number,
        "dates": list(dict.fromkeys(dates))[:10],
        "money_values": money[:20],
        "line_count": len(lines),
    }


def _safe_timings(conv_result) -> Dict[str, Any]:
    timings: Dict[str, Any] = {}
    if hasattr(conv_result, "timings") and isinstance(conv_result.timings, dict):
        for k, v in conv_result.timings.items():
            if hasattr(v, "times"):
                times = getattr(v, "times", None)
                if isinstance(times, list):
                    timings[k] = {
                        "count": len(times),
                        "min": min(times) if times else None,
                        "median": sorted(times)[len(times) // 2] if times else None,
                        "max": max(times) if times else None,
                    }
                else:
                    timings[k] = str(v)
            else:
                timings[k] = str(v)
    return timings


# ------------------------
# ROUTES
# ------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "pipeline": "docling_granite_258_remote",
        "hostport": OCR_HOSTPORT,
        "model": OCR_MODEL,
        "format": "DOCTAGS",
        "page_batch_size": PAGE_BATCH_SIZE,
    }


@app.post("/convert.ocr", response_model=OcrResult)
async def convert_ocr(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    _validate_pdf(file, pdf_bytes)

    filename = file.filename or "upload.pdf"
    doc_id = _make_doc_id(filename, pdf_bytes)

    ds = DocumentStream(name=filename, stream=BytesIO(pdf_bytes))

    try:
        conv_result = converter_ocr.convert(source=ds)
        doc = conv_result.document
        if not doc:
            raise RuntimeError("Docling conversion returned no document")

        markdown = doc.export_to_markdown()
        text = doc.export_to_text() if hasattr(doc, "export_to_text") else markdown

        num_pages = len(getattr(conv_result, "pages", []) or [])
        status = str(getattr(conv_result, "status", "UNKNOWN"))

        return OcrResult(
            doc_id=doc_id,
            filename=filename,
            status=status,
            num_pages=num_pages,
            pipeline="docling_granite_258_remote",
            timings=_safe_timings(conv_result),
            metadata=_extract_metadata(text),
            markdown=markdown,
            text=text,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR conversion error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn

    _log.info(
        "Starting OCR worker host=0.0.0.0 port=%s target=%s model=%s page_batch=%s",
        os.getenv("PORT", "8000"),
        OCR_HOSTPORT,
        OCR_MODEL,
        PAGE_BATCH_SIZE,
    )
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
