from __future__ import annotations

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main() -> None:
    # Input (inside container; mapped from ./data -> /app/data by docker-compose)
    input_doc_path = Path("/app/data/in/ocr-inputs/statement_sample1.pdf")
    if not input_doc_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_doc_path}")

    # Output (inside container; mapped from ./logs -> /app/logs by docker-compose)
    out_dir = Path("/app/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_doc_path.stem}.md"

    pipeline_options = PdfPipelineOptions()

    # NOTE: If your PDFs usually have a text layer, consider setting this False and only
    # enabling OCR for scanned docs. For now we keep it True as you requested.
    pipeline_options.do_ocr = True

    # Tables (requires OpenCV + libGL in the image)
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    # Use Tesseract CLI OCR (requires 'tesseract' binary installed in the image)
    # Keep lang tight for bank statements to improve accuracy and speed.
    ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    ocr_options.lang = "eng"
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    doc = converter.convert(input_doc_path).document
    md = doc.export_to_markdown()

    # Save to the mounted logs folder so it appears on the host at ./logs/<name>.md
    out_path.write_text(md, encoding="utf-8")

    # Still print to docker logs for quick inspection
    print(f"Saved markdown to: {out_path}")
    print(md)


if __name__ == "__main__":
    main()
