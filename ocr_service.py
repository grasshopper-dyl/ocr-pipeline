from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def _prov_key(prov: Dict[str, Any]) -> Tuple[int, float, float]:
    """
    Sort key: (page_no, -top_y, left_x)

    Docling bbox uses coord_origin = BOTTOMLEFT, so larger 't' means higher on the page.
    """
    page_no = int(prov.get("page_no", 0))
    bbox = prov.get("bbox") or {}
    t = float(bbox.get("t", 0.0))
    l = float(bbox.get("l", 0.0))
    return (page_no, -t, l)


def docling_dict_to_plain_text(payload: Dict[str, Any]) -> str:
    """
    Build a plain-text stream from Docling's 'texts' list, ordered by provenance.
    Avoids Docling's markdown/table serializer entirely.
    """
    texts: List[Dict[str, Any]] = payload.get("texts", []) or []

    extracted: List[Tuple[Tuple[int, float, float], str]] = []

    for t in texts:
        s = (t.get("text") or "").strip()
        if not s:
            continue

        # Many table renderings end up as '|' lines in markdown/text serializers.
        # In the dict, most real OCR words won't start with '|', so we can drop these safely.
        # (No regex; just a simple guard.)
        if s.startswith("|") and s.endswith("|"):
            continue

        prov_list = t.get("prov") or []
        if prov_list:
            key = _prov_key(prov_list[0])
        else:
            # If no prov, put at end
            key = (999999, 0.0, 0.0)

        extracted.append((key, s))

    extracted.sort(key=lambda x: x[0])

    # Simple paragraph-ish join: one item per line, collapse excessive blanks later.
    lines: List[str] = []
    prev_page = None
    for (page_no, _, _), s in extracted:
        if prev_page is None:
            prev_page = page_no
        elif page_no != prev_page:
            lines.append("")  # page break
            lines.append(f"--- PAGE {page_no} ---")
            lines.append("")
            prev_page = page_no

        # Minimal entity cleanup without html module
        s = s.replace("&amp;", "&")
        lines.append(s)

    # Collapse repeated blank lines
    out: List[str] = []
    blank = False
    for line in lines:
        if line.strip() == "":
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(line)
            blank = False

    return "\n".join(out).strip() + "\n"


def main() -> None:
    input_doc_path = Path("/app/data/in/ocr-inputs/statement_sample1.pdf")
    if not input_doc_path.exists():
        raise FileNotFoundError(f"PDF not found: {input_doc_path}")

    out_dir = Path("/app/logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True

    # For chunking: keep table structure OFF so nothing gets "consumed" into table objects
    pipeline_options.do_table_structure = False
    pipeline_options.table_structure_options.do_cell_matching = False

    ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    ocr_options.lang = ["eng"]
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    doc = converter.convert(input_doc_path).document
    payload = doc.export_to_dict()

    # Optional: keep JSON for debugging/provenance
    (out_dir / f"{input_doc_path.stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt = docling_dict_to_plain_text(payload)
    txt_path = out_dir / f"{input_doc_path.stem}.txt"
    txt_path.write_text(txt, encoding="utf-8")

    print(f"Saved plain text to: {txt_path}")
    print(txt)


if __name__ == "__main__":
    main()
