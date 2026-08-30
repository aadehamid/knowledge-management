[![Hugging Face's logo](/front/assets/huggingface_logo-noborder.svg) Hugging Face](/)

* [Models](/models)
* [Datasets](/datasets)
* [Spaces](/spaces)
* [Buckets new](/storage)
* [Docs](/docs)
* [Enterprise](/enterprise)
* [Pricing](/pricing)
* + Website

    - [Tasks](/tasks)
    - [HuggingChat](/chat)
    - [Collections](/collections)
    - [Languages](/languages)
    - [Organizations](/organizations)
  + Community

    - [Blog](/blog)
    - [Posts](/posts)
    - [Daily Papers](/papers)
    - [Hardware](/hardware)
    - [Learn](/learn)
    - [Discord](/join/discord)
    - [Forum](https://discuss.huggingface.co/)
    - [GitHub](https://github.com/huggingface)
  + Solutions

    - [Team & Enterprise](/enterprise)
    - [Hugging Face PRO](/pro)
    - [Enterprise Support](/support)
    - [Inference Providers](/inference/models)
    - [Inference Endpoints](/inference-endpoints)
    - [Storage Buckets](/storage)
* ---
* [Log In](/login)
* [Sign Up](/join)

# [![](https://cdn-avatars.huggingface.co/v1/production/uploads/1654942635336-5f3ff69679c1ba4c353d0c5a.png)](/PaddlePaddle) [PaddlePaddle](/PaddlePaddle) / [PaddleOCR-VL](/PaddlePaddle/PaddleOCR-VL) like 1.65k Follow ![](https://cdn-avatars.huggingface.co/v1/production/uploads/1654942635336-5f3ff69679c1ba4c353d0c5a.png) PaddlePaddle 3.95k

[Image-Text-to-Text](/models?pipeline_tag=image-text-to-text)[PaddleOCR](/models?library=PaddleOCR)[Safetensors](/models?library=safetensors)[English](/models?language=en)[Chinese](/models?language=zh)[multilingual](/models?language=multilingual)[paddleocr\_vl](/models?other=paddleocr_vl)[ERNIE4.5](/models?other=ERNIE4.5)[PaddlePaddle](/models?other=PaddlePaddle)[image-to-text](/models?other=image-to-text)[ocr](/models?other=ocr)[document-parse](/models?other=document-parse)[layout](/models?other=layout)[table](/models?other=table)[formula](/models?other=formula)[chart](/models?other=chart)[conversational](/models?other=conversational)[custom\_code](/models?other=custom_code)[Eval Results](/models?other=eval-results)

arxiv: 2510.14528

License: apache-2.0

[Model card](/PaddlePaddle/PaddleOCR-VL)  [Files Files and versions

xet](/PaddlePaddle/PaddleOCR-VL/tree/main)  [Community

97](/PaddlePaddle/PaddleOCR-VL/discussions)

Deploy

  Copy to bucket new

Use this model

### Instructions to use PaddlePaddle/PaddleOCR-VL with libraries, inference providers, notebooks, and local apps. Follow these links to get started.

* Libraries
 * [PaddleOCR](/PaddlePaddle/PaddleOCR-VL?library=PaddleOCR)

  How to use PaddlePaddle/PaddleOCR-VL with PaddleOCR:

  ```
  # See https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html to installation

  from paddleocr import PaddleOCRVL
  pipeline = PaddleOCRVL(pipeline_version="v1")
  output = pipeline.predict("path/to/document_image.png")
  for res in output:
  	res.print()
  	res.save_to_json(save_path="output")
  	res.save_to_markdown(save_path="output")
  ```
  * Notebooks
* [Google Colab](/PaddlePaddle/PaddleOCR-VL/colab)
* [Kaggle](/PaddlePaddle/PaddleOCR-VL/kaggle)
     * [Browse
  Quantizations](/models?other=base_model:quantized:PaddlePaddle/PaddleOCR-VL) to use this model in  llama.cpp,  Ollama,  LM Studio, or any compatible app.

A newer version of this model is available: [PaddlePaddle/PaddleOCR-VL-1.6](/PaddlePaddle/PaddleOCR-VL-1.6)

* [Introduction](#introduction "Introduction")
  + [**Core Features**](#core-features "Core Features")
  + [**Model Architecture**](#model-architecture "Model Architecture")
* [News](#news "News")
* [Usage](#usage "Usage")
  + [Install Dependencies](#install-dependencies "Install Dependencies")
  + [Basic Usage](#basic-usage "Basic Usage")
  + [Accelerate VLM Inference via Optimized Inference Servers](#accelerate-vlm-inference-via-optimized-inference-servers "Accelerate VLM Inference via Optimized Inference Servers")
* [PaddleOCR-VL-0.9B Usage with transformers](#paddleocr-vl-09b-usage-with-transformers "PaddleOCR-VL-0.9B Usage with transformers")
* [Performance](#performance "Performance")
  + [Page-Level Document Parsing](#page-level-document-parsing "Page-Level Document Parsing")
    - [1. OmniDocBench v1.5](#1-omnidocbench-v15 "1. OmniDocBench v1.5")
    - [2. OmniDocBench v1.0](#2-omnidocbench-v10 "2. OmniDocBench v1.0")
  + [Element-level Recognition](#element-level-recognition "Element-level Recognition")
    - [1. Text](#1-text "1. Text")
    - [2. Table](#2-table "2. Table")
    - [3. Formula](#3-formula "3. Formula")
    - [4. Chart](#4-chart "4. Chart")
* [Visualization](#visualization "Visualization")
  + [Comprehensive Document Parsing](#comprehensive-document-parsing "Comprehensive Document Parsing")
  + [Text](#text "Text")
  + [Table](#table "Table")
  + [Formula](#formula "Formula")
  + [Chart](#chart "Chart")
* [Acknowledgments](#acknowledgments "Acknowledgments")
* [Citation](#citation "Citation")

# PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model

[![repo](https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf)](https://github.com/PaddlePaddle/PaddleOCR)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-black.svg?logo=data:image/png;base64,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&labelColor=white)](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
[![ModelScope](https://img.shields.io/badge/ModelScope-black?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIzIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCiA8Zz4KICA8dGl0bGU+TGF5ZXIgMTwvdGl0bGU+CiAgPHBhdGggaWQ9InN2Z18xNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTAsODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTUiIGZpbGw9IiM2MjRhZmYiIGQ9Im05OS4xNCwxMTUuNDlsMjUuNjUsMGwwLDI1LjY1bC0yNS42NSwwbDAsLTI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTYiIGZpbGw9IiM2MjRhZmYiIGQ9Im0xNzYuMDksMTQxLjE0bC0yNS42NDk5OSwwbDAsMjIuMTlsNDcuODQsMGwwLC00Ny44NGwtMjIuMTksMGwwLDI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTciIGZpbGw9IiMzNmNmZDEiIGQ9Im0xMjQuNzksODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTgiIGZpbGw9IiMzNmNmZDEiIGQ9Im0wLDY0LjE5bDI1LjY1LDBsMCwyNS42NWwtMjUuNjUsMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzE5IiBmaWxsPSIjNjI0YWZmIiBkPSJtMTk4LjI4LDg5Ljg0bDI1LjY0OTk5LDBsMCwyNS42NDk5OWwtMjUuNjQ5OTksMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIwIiBmaWxsPSIjMzZjZmQxIiBkPSJtMTk4LjI4LDY0LjE5bDI1LjY0OTk5LDBsMCwyNS42NWwtMjUuNjQ5OTksMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIxIiBmaWxsPSIjNjI0YWZmIiBkPSJtMTUwLjQ0LDQybDAsMjIuMTlsMjUuNjQ5OTksMGwwLDI1LjY1bDIyLjE5LDBsMCwtNDcuODRsLTQ3Ljg0LDB6Ii8+CiAgPHBhdGggaWQ9InN2Z18yMiIgZmlsbD0iIzM2Y2ZkMSIgZD0ibTczLjQ5LDg5Ljg0bDI1LjY1LDBsMCwyNS42NDk5OWwtMjUuNjUsMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIzIiBmaWxsPSIjNjI0YWZmIiBkPSJtNDcuODQsNjQuMTlsMjUuNjUsMGwwLC0yMi4xOWwtNDcuODQsMGwwLDQ3Ljg0bDIyLjE5LDBsMCwtMjUuNjV6Ii8+CiAgPHBhdGggaWQ9InN2Z18yNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTQ3Ljg0LDExNS40OWwtMjIuMTksMGwwLDQ3Ljg0bDQ3Ljg0LDBsMCwtMjIuMTlsLTI1LjY1LDBsMCwtMjUuNjV6Ii8+CiA8L2c+Cjwvc3ZnPg==&labelColor=white)](https://modelscope.cn/models/PaddlePaddle/PaddleOCR-VL)
[![HuggingFace](https://img.shields.io/badge/Demo_on_HuggingFace-black.svg?logo=data:image/png;base64,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&labelColor=white)](https://huggingface.co/spaces/PaddlePaddle/PaddleOCR-VL_Online_Demo)
[![ModelScope](https://img.shields.io/badge/Demo_on_ModelScope-black?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIzIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCiA8Zz4KICA8dGl0bGU+TGF5ZXIgMTwvdGl0bGU+CiAgPHBhdGggaWQ9InN2Z18xNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTAsODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTUiIGZpbGw9IiM2MjRhZmYiIGQ9Im05OS4xNCwxMTUuNDlsMjUuNjUsMGwwLDI1LjY1bC0yNS42NSwwbDAsLTI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTYiIGZpbGw9IiM2MjRhZmYiIGQ9Im0xNzYuMDksMTQxLjE0bC0yNS42NDk5OSwwbDAsMjIuMTlsNDcuODQsMGwwLC00Ny44NGwtMjIuMTksMGwwLDI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTciIGZpbGw9IiMzNmNmZDEiIGQ9Im0xMjQuNzksODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTgiIGZpbGw9IiMzNmNmZDEiIGQ9Im0wLDY0LjE5bDI1LjY1LDBsMCwyNS42NWwtMjUuNjUsMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzE5IiBmaWxsPSIjNjI0YWZmIiBkPSJtMTk4LjI4LDg5Ljg0bDI1LjY0OTk5LDBsMCwyNS42NDk5OWwtMjUuNjQ5OTksMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIwIiBmaWxsPSIjMzZjZmQxIiBkPSJtMTk4LjI4LDY0LjE5bDI1LjY0OTk5LDBsMCwyNS42NWwtMjUuNjQ5OTksMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIxIiBmaWxsPSIjNjI0YWZmIiBkPSJtMTUwLjQ0LDQybDAsMjIuMTlsMjUuNjQ5OTksMGwwLDI1LjY1bDIyLjE5LDBsMCwtNDcuODRsLTQ3Ljg0LDB6Ii8+CiAgPHBhdGggaWQ9InN2Z18yMiIgZmlsbD0iIzM2Y2ZkMSIgZD0ibTczLjQ5LDg5Ljg0bDI1LjY1LDBsMCwyNS42NDk5OWwtMjUuNjUsMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIzIiBmaWxsPSIjNjI0YWZmIiBkPSJtNDcuODQsNjQuMTlsMjUuNjUsMGwwLC0yMi4xOWwtNDcuODQsMGwwLDQ3Ljg0bDIyLjE5LDBsMCwtMjUuNjV6Ii8+CiAgPHBhdGggaWQ9InN2Z18yNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTQ3Ljg0LDExNS40OWwtMjIuMTksMGwwLDQ3Ljg0bDQ3Ljg0LDBsMCwtMjIuMTlsLTI1LjY1LDBsMCwtMjUuNjV6Ii8+CiA8L2c+Cjwvc3ZnPg==&labelColor=white)](https://modelscope.cn/studios/PaddlePaddle/PaddleOCR-VL_Online_Demo/summary)
[![Discord](https://img.shields.io/badge/Discord-ERNIE-5865F2?logo=discord&logoColor=white)](https://discord.gg/JPmZXDsEEK)
[![X](https://img.shields.io/badge/X-PaddlePaddle-6080F0)](https://x.com/PaddlePaddle)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](/PaddlePaddle/PaddleOCR-VL/tree/main/LICENSE)

**🔥 Official Website**: [Baidu AI Studio](https://aistudio.baidu.com/paddleocr) |
**📝 arXiv**: [Technical Report](https://arxiv.org/pdf/2510.14528)

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/allmetric.png)

## Introduction

**PaddleOCR-VL** is a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition. This innovative model efficiently supports 109 languages and excels in recognizing complex elements (e.g., text, tables, formulas, and charts), while maintaining minimal resource consumption. Through comprehensive evaluations on widely used public benchmarks and in-house benchmarks, PaddleOCR-VL achieves SOTA performance in both page-level document parsing and element-level recognition. It significantly outperforms existing solutions, exhibits strong competitiveness against top-tier VLMs, and delivers fast inference speeds. These strengths make it highly suitable for practical deployment in real-world scenarios.

### **Core Features**

1. **Compact yet Powerful VLM Architecture:** We present a novel vision-language model that is specifically designed for resource-efficient inference, achieving outstanding performance in element recognition. By integrating a NaViT-style dynamic high-resolution visual encoder with the lightweight ERNIE-4.5-0.3B language model, we significantly enhance the model’s recognition capabilities and decoding efficiency. This integration maintains high accuracy while reducing computational demands, making it well-suited for efficient and practical document processing applications.
2. **SOTA Performance on Document Parsing:** PaddleOCR-VL achieves state-of-the-art performance in both page-level document parsing and element-level recognition. It significantly outperforms existing pipeline-based solutions and exhibiting strong competitiveness against leading vision-language models (VLMs) in document parsing. Moreover, it excels in recognizing complex document elements, such as text, tables, formulas, and charts, making it suitable for a wide range of challenging content types, including handwritten text and historical documents. This makes it highly versatile and suitable for a wide range of document types and scenarios.
3. **Multilingual Support:** PaddleOCR-VL Supports 109 languages, covering major global languages, including but not limited to Chinese, English, Japanese, Latin, and Korean, as well as languages with different scripts and structures, such as Russian (Cyrillic script), Arabic, Hindi (Devanagari script), and Thai. This broad language coverage substantially enhances the applicability of our system to multilingual and globalized document processing scenarios.

### **Model Architecture**

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/paddleocrvl.png)

## News

* `2025.11.07` 🚀 Enabled `flash-attn` in the `transformers` library to achieve faster inference with PaddleOCR-VL-0.9B.
* `2025.11.04` 🌟 PaddleOCR-VL-0.9B is now officially supported on `vLLM` .
* `2025.10.29` 🤗 Supports calling the core module PaddleOCR-VL-0.9B of PaddleOCR-VL via the `transformers` library.
* `2025.10.16` 🚀 We release [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR), — a multilingual documents parsing via a 0.9B Ultra-Compact Vision-Language Model with SOTA performance.

## Usage

### Install Dependencies

Install [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR):

```
# The following command installs the PaddlePaddle version for CUDA 12.6. For other CUDA versions and the CPU version, please refer to https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]>=3.4.0"
```

### Basic Usage

CLI usage:

```
paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png --pipeline_version v1
```

Python API usage:

```
from paddleocr import PaddleOCRVL
pipeline = PaddleOCRVL(pipeline_version="v1")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png")
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
```

### Accelerate VLM Inference via Optimized Inference Servers

1. Start the VLM inference server:

   You can start the vLLM inference service using one of two methods:

   * Method 1: PaddleOCR method

     ```
     docker run \
         --rm \
         --gpus all \
         --network host \
         ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
         paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8080 --backend vllm
     ```
   * Method 2: vLLM method

     [vLLM: PaddleOCR-VL Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html)
2. Call the PaddleOCR CLI or Python API:

   ```
   paddleocr doc_parser \
       -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png \
       --pipeline_version v1 \
       --vl_rec_backend vllm-server \
       --vl_rec_server_url http://127.0.0.1:8080/v1
   ```

   ```
   from paddleocr import PaddleOCRVL
   pipeline = PaddleOCRVL(pipeline_version="v1", vl_rec_backend="vllm-server", vl_rec_server_url="http://127.0.0.1:8080/v1")
   output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png")
   for res in output:
       res.print()
       res.save_to_json(save_path="output")
       res.save_to_markdown(save_path="output")
   ```

**For more usage details and parameter explanations, see the [documentation](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html).**

## PaddleOCR-VL-0.9B Usage with transformers

Currently, we support inference using the PaddleOCR-VL-0.9B model with the `transformers` library, which can recognize texts, formulas, tables, and chart elements. In the future, we plan to support full document parsing inference with `transformers`. Below is a simple script we provide to support inference using the PaddleOCR-VL-0.9B model with `transformers`.

> Note: We currently recommend using the official method for inference, as it is faster and supports page-level document parsing. The example code below only supports element-level recognition.

```
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# ---- Settings ----
model_path = "PaddlePaddle/PaddleOCR-VL"
image_path = "test.png"
task = "ocr" # Options: 'ocr' | 'table' | 'chart' | 'formula'
# ------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

image = Image.open(image_path).convert("RGB")

model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {"role": "user",
     "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[task]},
        ]
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(DEVICE)

outputs = model.generate(**inputs, max_new_tokens=1024)
outputs = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(outputs)
```

👉 Click to expand: Use flash-attn to boost performance and reduce memory usage

```
# ensure the flash-attn2 is installed
pip install flash-attn --no-build-isolation
```

```
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

# ---- Settings ----
model_path = "PaddlePaddle/PaddleOCR-VL"
image_path = "test.png"
task = "ocr" # ← change to "table" | "chart" | "formula"
# ------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(dtype=torch.bfloat16, device=DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
}
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text",  "text": PROMPTS[task]}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(DEVICE)

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        use_cache=True
    )

outputs = processor.batch_decode(out, skip_special_tokens=True)[0]
print(outputs)
```

## Performance

### Page-Level Document Parsing

#### 1. OmniDocBench v1.5

##### PaddleOCR-VL achieves SOTA performance for overall, text, formula, tables and reading order on OmniDocBench v1.5

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/omni15.png)

#### 2. OmniDocBench v1.0

##### PaddleOCR-VL achieves SOTA performance for almost all metrics of overall, text, formula, tables and reading order on OmniDocBench v1.0

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/omni10.png)

> **Notes:**
>
> * The metrics are from [MinerU](https://github.com/opendatalab/MinerU), [OmniDocBench](https://github.com/opendatalab/OmniDocBench), and our own internal evaluations.

### Element-level Recognition

#### 1. Text

**Comparison of OmniDocBench-OCR-block Performance**

PaddleOCR-VL’s robust and versatile capability in handling diverse document types, establishing it as the leading method in the OmniDocBench-OCR-block performance evaluation.

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/omnibenchocr.png)

**Comparison of In-house-OCR Performance**

In-house-OCR provides a evaluation of performance across multiple languages and text types. Our model demonstrates outstanding accuracy with the lowest edit distances in all evaluated scripts.

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/inhouseocr.png)

#### 2. Table

**Comparison of In-house-Table Performance**

Our self-built evaluation set contains diverse types of table images, such as Chinese, English, mixed Chinese-English, and tables with various characteristics like full, partial, or no borders, book/manual formats, lists, academic papers, merged cells, as well as low-quality, watermarked, etc. PaddleOCR-VL achieves remarkable performance across all categories.

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/inhousetable.png)

#### 3. Formula

**Comparison of In-house-Formula Performance**

In-house-Formula evaluation set contains simple prints, complex prints, camera scans, and handwritten formulas. PaddleOCR-VL demonstrates the best performance in every category.

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/inhouse-formula.png)

#### 4. Chart

**Comparison of In-house-Chart Performance**

The evaluation set is broadly categorized into 11 chart categories, including bar-line hybrid, pie, 100% stacked bar, area, bar, bubble, histogram, line, scatterplot, stacked area, and stacked bar. PaddleOCR-VL not only outperforms expert OCR VLMs but also surpasses some 72B-level multimodal language models.

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/inhousechart.png)

## Visualization

### Comprehensive Document Parsing

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/overview1.jpg)
![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/overview2.jpg)
![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/overview3.jpg)
![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/overview4.jpg)

### Text

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/text_english_arabic.jpg)
![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/text_handwriting_02.jpg)

### Table

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/table_01.jpg)
![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/table_02.jpg)

### Formula

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/formula_EN.jpg)
![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/formula_ZH.jpg)

### Chart

![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/chart_01.jpg)
![](https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/chart_02.jpg)

## Acknowledgments

We would like to thank [ERNIE](https://github.com/PaddlePaddle/ERNIE), [Keye](https://github.com/Kwai-Keye/Keye), [MinerU](https://github.com/opendatalab/MinerU), [OmniDocBench](https://github.com/opendatalab/OmniDocBench) for providing valuable code, model weights and benchmarks. We also appreciate everyone's contribution to this open-source project!

## Citation

If you find PaddleOCR-VL helpful, feel free to give us a star and citation.

```
@misc{cui2025paddleocrvlboostingmultilingualdocument,
      title={PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model},
      author={Cheng Cui and Ting Sun and Suyin Liang and Tingquan Gao and Zelun Zhang and Jiaxuan Liu and Xueqing Wang and Changda Zhou and Hongen Liu and Manhui Lin and Yue Zhang and Yubo Zhang and Handong Zheng and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2025},
      eprint={2510.14528},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.14528},
}
```

Downloads last month
:   8,958

Safetensors

Model size

1.0B params

Tensor type

BF16

·

Chat template

Files info

Inference Providers [NEW](https://huggingface.co/docs/inference-providers)

[Image-Text-to-Text](/tasks/image-text-to-text "Learn more about image-text-to-text")

This model isn't deployed by any Inference Provider. [🙋 59 Ask for provider support](/spaces/huggingface/InferenceSupport/discussions/5558)

## Model tree for PaddlePaddle/PaddleOCR-VL

Base model

[baidu/ERNIE-4.5-0.3B-Paddle](/baidu/ERNIE-4.5-0.3B-Paddle)

Finetuned

 ([27](/models?other=base_model:finetune:baidu/ERNIE-4.5-0.3B-Paddle))

this model

Adapters

 [7 models](/models?other=base_model:adapter:PaddlePaddle/PaddleOCR-VL)

Finetunes

 [5 models](/models?other=base_model:finetune:PaddlePaddle/PaddleOCR-VL)

Merges

 [3 models](/models?other=base_model:merge:PaddlePaddle/PaddleOCR-VL)

Quantizations

 [6 models](/models?other=base_model:quantized:PaddlePaddle/PaddleOCR-VL)

## Spaces using PaddlePaddle/PaddleOCR-VL 26

[![](https://cdn-avatars.huggingface.co/v1/production/uploads/1654942635336-5f3ff69679c1ba4c353d0c5a.png)

PaddlePaddle/PaddleOCR-VL\_Online\_Demo](/spaces/PaddlePaddle/PaddleOCR-VL_Online_Demo) [📚

eduagarcia/multilingual-tokenizer-leaderboard](/spaces/eduagarcia/multilingual-tokenizer-leaderboard) [🔎

Upsampler/paddleocr-vl](/spaces/Upsampler/paddleocr-vl) [📈

eoeooe/PaddleOCR-VL-Request](/spaces/eoeooe/PaddleOCR-VL-Request) [🛡️

cronos3k/document-integrity-verifier](/spaces/cronos3k/document-integrity-verifier) [📄

Tutel314/PaddleOCR-VL-Demo](/spaces/Tutel314/PaddleOCR-VL-Demo) [📄

studentoflife/paddleocr-vl-demo](/spaces/studentoflife/paddleocr-vl-demo) [📈

markobinario/PaddleOCR-VL\_Online\_Demo](/spaces/markobinario/PaddleOCR-VL_Online_Demo)  + 21 Spaces + 18 Spaces

## Collection including PaddlePaddle/PaddleOCR-VL

[#### PaddleOCR-VL

Collection

Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model • 5 items • Updated Feb 11 •  32](/collections/PaddlePaddle/paddleocr-vl)

## Paper for PaddlePaddle/PaddleOCR-VL

[#### PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model

Paper • 2510.14528 • Published Oct 16, 2025 •  129](/papers/2510.14528)

## Evaluation results

* [Delores-Lin/MDPBench](/datasets/Delores-Lin/MDPBench)  [leaderboard](/datasets/Delores-Lin/MDPBench?eval_result=PaddlePaddle/PaddleOCR-VL)
* Overall [View evaluation results](/PaddlePaddle/PaddleOCR-VL/discussions/98)  [![](https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/LgZxNkDzfdr9DP6tqb8bh.png)

   source](https://huggingface.co/datasets/Delores-Lin/MDPBench)

  69.6
* Digital [View evaluation results](/PaddlePaddle/PaddleOCR-VL/discussions/98)  [![](https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/LgZxNkDzfdr9DP6tqb8bh.png)

   source](https://huggingface.co/datasets/Delores-Lin/MDPBench)

  87.6
* Photographed [View evaluation results](/PaddlePaddle/PaddleOCR-VL/discussions/98)  [![](https://cdn-avatars.huggingface.co/v1/production/uploads/no-auth/LgZxNkDzfdr9DP6tqb8bh.png)

   source](https://huggingface.co/datasets/Delores-Lin/MDPBench)

  63.6
 * +20 more
* [allenai/olmOCR-bench](/datasets/allenai/olmOCR-bench)  [leaderboard](/datasets/allenai/olmOCR-bench?eval_result=PaddlePaddle/PaddleOCR-VL)
* Overall [View evaluation results](/PaddlePaddle/PaddleOCR-VL/discussions/89)  [![](https://cdn-avatars.huggingface.co/v1/production/uploads/1608042047613-5f1158120c833276f61f1a84.jpeg)

   source](https://huggingface.co/papers/2510.14528)

  80
 * +7 more

Expand 2 benchmarks

System theme

Company

[TOS](/terms-of-service) [Privacy](/privacy) [About](/huggingface) [Careers](https://apply.workable.com/huggingface/)

Website

[Models](/models) [Datasets](/datasets) [Spaces](/spaces) [Pricing](/pricing) [Docs](/docs)