from __future__ import annotations

from pathlib import Path
import json

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main() -> None:
    input_doc_path = Path("/app/data/in/ocr-inputs/statement_sample1.pdf")
    if not input_doc_path.exists():
        raise FileNotFoundError(f"PDF not found: {input_doc_path}")

    out_dir = Path("/app/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_doc_path.stem}.json"

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True

    # NOTE: If you turn this on and it "cuts out" text, that's table reconstruction
    # consuming text blocks. Keep it off unless you specifically need tables.
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    ocr_options.lang = ["eng"]
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    doc = converter.convert(input_doc_path).document

    # Docling returns a Python dict (Docling-native)
    payload = doc.export_to_dict()

    # Write as JSON
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved json to: {out_path}")
    # Optional: print only a small preview so docker logs don't explode
    print(payload)


if __name__ == "__main__":
    main()
