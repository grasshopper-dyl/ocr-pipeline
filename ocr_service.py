from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


# ------------------------------------------------------------
# Plain-text reconstruction from OCR tokens (non-table content)
# ------------------------------------------------------------
def extract_text_lines(payload: Dict[str, Any]) -> List[str]:
    texts = payload.get("texts", [])
    tokens: List[Tuple[int, float, float, str]] = []
    heights: List[float] = []

    for t in texts:
        s = (t.get("text") or "").strip()
        if not s:
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

        y = (t_ + b) / 2
        h = t_ - b

        tokens.append((page, y, l, s))
        heights.append(h)

    tokens.sort(key=lambda x: (x[0], -x[1], x[2]))

    if not heights:
        return []

    heights.sort()
    med_h = heights[len(heights) // 2]
    y_tol = max(2.0, med_h * 0.6)

    lines: List[str] = []
    current: List[str] = []
    current_y = None
    current_page = None

    for page, y, x, s in tokens:
        if current_page is None:
            current_page = page

        if page != current_page:
            if current:
                lines.append(" ".join(current))
            lines.append("")
            lines.append(f"--- PAGE {page} ---")
            lines.append("")
            current = []
            current_y = None
            current_page = page

        if current_y is None or abs(y - current_y) <= y_tol:
            current.append(s)
            current_y = y if current_y is None else current_y
        else:
            lines.append(" ".join(current))
            current = [s]
            current_y = y

    if current:
        lines.append(" ".join(current))

    return lines


# ------------------------------------------------------------
# Table extraction (REAL tables, no OCR guessing)
# ------------------------------------------------------------
def extract_tables(payload: Dict[str, Any]) -> List[str]:
    out: List[str] = []

    for table in payload.get("tables", []):
        data = table.get("data") or {}
        grid = data.get("grid") or []

        if not grid:
            continue

        out.append("")
        out.append("=== TABLE ===")

        for row in grid:
            cells = []
            for cell in row:
                text = (cell.get("text") or "").strip()
                cells.append(text)

            # Pipe-separated plain text (easy to parse later)
            out.append(" | ".join(cells))

        out.append("=== END TABLE ===")
        out.append("")

    return out


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    input_pdf = Path("/app/data/in/ocr-inputs/statement_sample1.pdf")
    if not input_pdf.exists():
        raise FileNotFoundError(input_pdf)

    out_dir = Path("/app/logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Docling pipeline
    pipeline = PdfPipelineOptions()
    pipeline.do_ocr = True
    pipeline.do_table_structure = False  # <-- ENABLE REAL TABLE PARSING

    ocr = TesseractCliOcrOptions(force_full_page_ocr=True)
    ocr.lang = ["eng"]
    pipeline.ocr_options = ocr

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)
        }
    )

    doc = converter.convert(input_pdf).document
    payload = doc.export_to_dict()

    # Save raw JSON
    (out_dir / f"{input_pdf.stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Build output
    lines = extract_text_lines(payload)
    tables = extract_tables(payload)

    full_text = "\n".join(lines + tables).strip() + "\n"

    out_txt = out_dir / f"{input_pdf.stem}.txt"
    out_txt.write_text(full_text, encoding="utf-8")

    print(f"Saved plain text to: {out_txt}")
    print(full_text)


if __name__ == "__main__":
    main()
