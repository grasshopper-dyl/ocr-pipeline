# Prerequisites:
# pip install vllm
# pip install docling_core
# place page images you want to convert into "img/" dir

import time
import os
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from pathlib import Path

# Configuration
MODEL_PATH = "ibm-granite/granite-docling-258M"
IMAGE_DIR = "pdf/"  # Place your page images here
OUTPUT_DIR = "out/"
PROMPT_TEXT = "Convert this page to docling."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT_TEXT},
        ],
    },
]


# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize LLM
llm = LLM(model=MODEL_PATH, revision="untied", limit_mm_per_prompt={"image": 1})
processor = AutoProcessor.from_pretrained(MODEL_PATH)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    skip_special_tokens=False,
)

# Load and prepare all images and prompts up front
batched_inputs = []
image_names = []

for img_file in sorted(os.listdir(IMAGE_DIR)):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(IMAGE_DIR, img_file)
        with Image.open(img_path) as im:
            image = im.convert("RGB")

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        batched_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        image_names.append(os.path.splitext(img_file)[0])

# Run batch inference
start_time = time.time()
outputs = llm.generate(batched_inputs, sampling_params=sampling_params)

# Postprocess all results
for img_fn, output, input_data in zip(image_names, outputs, batched_inputs):
    doctags = output.outputs[0].text
    output_path_dt = Path(OUTPUT_DIR) / f"{img_fn}.dt"
    output_path_md = Path(OUTPUT_DIR) / f"{img_fn}.md"

    with open(output_path_dt, "w", encoding="utf-8") as f:
        f.write(doctags)

    # Convert to DoclingDocument and save markdown
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [input_data["multi_modal_data"]["image"]])
    doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
    doc.save_as_markdown(output_path_md)

print(f"Total time: {time.time() - start_time:.2f} sec")