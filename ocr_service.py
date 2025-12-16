from __future__ import annotations

from pathlib import Path
import html

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main() -> None:
    input_doc_path = Path("/app/data/in/ocr-inputs/statement_sample1.pdf")
    if not input_doc_path.exists():
        raise FileNotFoundError(f"PDF not found: {input_doc_path}")

    out_dir = Path("/app/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_doc_path.stem}.md"

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True

    # OFF for readability (prevents the “everything in one table cell” output)
    pipeline_options.do_table_structure = False
    pipeline_options.table_structure_options.do_cell_matching = False

    ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    ocr_options.lang = ["eng"]
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    doc = converter.convert(input_doc_path).document
    md = html.unescape(doc.export_to_markdown())

    out_path.write_text(md, encoding="utf-8")
    print(f"Saved markdown to: {out_path}")
    print(md)


if __name__ == "__main__":
    main()
