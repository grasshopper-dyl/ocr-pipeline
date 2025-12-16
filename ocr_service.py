from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def docling_plain_text_lines(payload: Dict[str, Any]) -> str:
    """
    Convert Docling payload to readable plain text by reconstructing lines
    from word-level OCR tokens using Y-position grouping.
    """

    texts = payload.get("texts", [])
    if not texts:
        return ""

    tokens: List[Tuple[int, float, float, str]] = []
    heights: List[float] = []

    for t in texts:
        text = (t.get("text") or "").strip()
        if not text:
            continue

        prov = (t.get("prov") or [{}])[0]
        bbox = prov.get("bbox") or {}

        try:
            page = int(prov.get("page_no", 0))
            l = float(bbox["l"])
            b = float(bbox["b"])
            t_ = float(bbox["t"])
        except Exception:
            continue

        y_center = (t_ + b) / 2
        height = t_ - b

        tokens.append((page, y_center, l, text))
        heights.append(height)

    # Sort: page → top-to-bottom → left-to-right
    tokens.sort(key=lambda x: (x[0], -x[1], x[2]))

    # Median token height → line tolerance
    heights.sort()
    med_h = heights[len(heights) // 2] if heights else 10.0
    y_tol = max(2.0, med_h * 0.6)

    lines: List[str] = []
    current_line: List[str] = []
    current_y = None
    current_page = None

    for page, y, x, text in tokens:
        if current_page is None:
            current_page = page

        # Page break
        if page != current_page:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = []

            lines.append("")
            lines.append(f"--- PAGE {page} ---")
            lines.append("")
            current_page = page
            current_y = None

        if current_y is None or abs(y - current_y) <= y_tol:
            current_line.append(text)
            current_y = y if current_y is None else current_y
        else:
            lines.append(" ".join(current_line))
            current_line = [text]
            current_y = y

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    input_pdf = Path("/app/data/in/ocr-inputs/statement_sample1.pdf")
    if not input_pdf.exists():
        raise FileNotFoundError(f"PDF not found: {input_pdf}")

    out_dir = Path("/app/logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Docling configuration
    # -------------------------
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True

    # IMPORTANT: keep table reconstruction OFF
    pipeline_options.do_table_structure = False

    ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    ocr_options.lang = ["eng"]
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert document
    doc = converter.convert(input_pdf).document
    payload = doc.export_to_dict()

    # Save raw JSON (debug / provenance)
    json_path = out_dir / f"{input_pdf.stem}.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Extract clean plain text
    text = docling_plain_text_lines(payload)
    txt_path = out_dir / f"{input_pdf.stem}.txt"
    txt_path.write_text(text, encoding="utf-8")

    print(f"Saved plain text to: {txt_path}")
    print(text)


if __name__ == "__main__":
    main()
