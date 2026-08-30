title: allenai/olmOCR-2-7B-1025-FP8 · Hugging Face
description: We’re on a journey to advance and democratize artificial intelligence through open source and open science.

![olmOCR Logo](https://cdn-uploads.huggingface.co/production/uploads/6734d6722769638944a5aa2e/DPsr3ZvRF9v-gdMa4EaHW.png){width=300px}

#  olmOCR-2-7B-1025-FP8 

Quantized to FP8 Version of [olmOCR-2-7B-1025](https://huggingface.co/allenai/olmOCR-2-7B-1025), using llmcompressor.

This is a release of the olmOCR model that's fine tuned from Qwen2.5-VL-7B-Instruct using the [olmOCR-mix-1025](https://huggingface.co/datasets/allenai/olmOCR-mix-1025) dataset. It has been additionally fine tuned using GRPO RL training to boost its performance at math equations, tables, and other tricky OCR cases.

Quick links:

- 📃 [Paper](https://olmocr.allenai.org/papers/olmocr.pdf)
- 🤗 [SFT Dataset](https://huggingface.co/datasets/allenai/olmOCR-mix-1025)
- 🤗 [RL Dataset](https://huggingface.co/datasets/allenai/olmOCR-synthmix-1025)
- 🛠️ [Code](https://github.com/allenai/olmocr)
- 🎮 [Demo](https://olmocr.allenai.org/)

The best way to use this model is via the [olmOCR toolkit](https://github.com/allenai/olmocr). The toolkit comes with an efficient inference setup via VLLM that can handle millions of documents at scale.

##  olmOCR-Bench Scores 

This model scores the following scores on [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench) when used with the [olmOCR toolkit](https://github.com/allenai/olmocr) toolkit which automatically renders, rotates, and retries pages as needed.

| **Model** | ArXiv | Old Scans Math | Tables | Old Scans | Headers and Footers | Multi column | Long tiny text | Base | Overall |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| olmOCR pipeline v0.4.0 with olmOCR-2-7B-1025 | 82.9 | 82.1 | 84.3 | 48.3 | 95.7 | 84.3 | 81.4 | 99.7 | 82.3 ± 1.1 |
| olmOCR pipeline v0.4.0 with olmOCR-2-7B-1025-FP8 | 83.0 | 82.3 | 84.9 | 47.7 | 96.1 | 83.7 | 81.9 | 99.7 | 82.4 ± 1.1 |

##  Usage 

This model expects as input a single document image, rendered such that the longest dimension is 1288 pixels.

The prompt must then contain the additional metadata from the document, and the easiest way to generate this is to use the methods provided by the [olmOCR toolkit](https://github.com/allenai/olmocr).

##  Manual Prompting 

If you want to prompt this model manually instead of using the [olmOCR toolkit](https://github.com/allenai/olmocr), please see the code below.

In normal usage, the olmOCR toolkit builds the prompt by rendering the PDF page, and extracting relevant text blocks and image metadata. To duplicate that you will need to

```bash
pip install olmocr>=0.4.0
```

and then run the following sample code.

```python
import torch
import base64
import urllib.request

from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

# Initialize the model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("allenai/olmOCR-2-7B-1025-FP8", device_map="auto").eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Grab a sample PDF
urllib.request.urlretrieve("https://olmocr.allenai.org/papers/olmocr.pdf", "./paper.pdf")

# Render page 1 to an image
image_base64 = render_pdf_to_base64png("./paper.pdf", 1, target_longest_image_dim=1288)


# Build the full prompt
messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]

# Apply the chat template and processor
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

inputs = processor(
    text=[text],
    images=[main_image],
    padding=True,
    return_tensors="pt",
)
inputs = {key: value.to(device) for (key, value) in inputs.items()}


# Generate the output
output = model.generate(
            **inputs,
            temperature=0.1,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=True,
        )

# Decode the output
prompt_length = inputs["input_ids"].shape[1]
new_tokens = output[:, prompt_length:]
text_output = processor.tokenizer.batch_decode(
    new_tokens, skip_special_tokens=True
)

print(text_output)
# ['---\nprimary_language: en\nis_rotation_valid: True\nrotation_correction: 0\nis_table: False\nis_diagram: False\n---\nolmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models\n\nJake Poz']
```

##  License and use 

This model is licensed under Apache 2.0. It is intended for research and educational use in accordance with Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use).
