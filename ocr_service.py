import sys
import torch
from pathlib import Path

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IN_DIR = Path("/app/data/in")
OUT_DIR = Path("/app/data/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python ocr_service.py <filename_in_/app/data/in>")
    sys.exit(1)

input_path = IN_DIR / sys.argv[1]
if not input_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

# Load image from local file in the container
image = load_image(f"file://{input_path}")

processor = AutoProcessor.from_pretrained("ibm-granite/granite-docling-258M")
model = AutoModelForVision2Seq.from_pretrained(
    "ibm-granite/granite-docling-258M",
    dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    attn_implementation="sdpa",   # explicitly NOT flash_attention_2
).to(DEVICE)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Convert this page to docling."},
        ],
    },
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)

generated_ids = model.generate(**inputs, max_new_tokens=8192)
prompt_len = inputs.input_ids.shape[1]
trimmed = generated_ids[:, prompt_len:]

doctags = processor.batch_decode(trimmed, skip_special_tokens=False)[0].lstrip()

doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
doc = DoclingDocument.load_from_doctags(doctags_doc, document_name=input_path.stem)

# Write outputs to mounted folder
(OUT_DIR / f"{input_path.stem}.doctags.txt").write_text(doctags, encoding="utf-8")
(OUT_DIR / f"{input_path.stem}.md").write_text(doc.export_to_markdown(), encoding="utf-8")

print(f"Wrote:\n- {OUT_DIR / (input_path.stem + '.doctags.txt')}\n- {OUT_DIR / (input_path.stem + '.md')}")
