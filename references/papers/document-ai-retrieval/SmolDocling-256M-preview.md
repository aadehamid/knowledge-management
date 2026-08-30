title: docling-project/SmolDocling-256M-preview · Hugging Face
description: We’re on a journey to advance and democratize artificial intelligence through open source and open science.

# docling-project/SmolDocling-256M-preview · Hugging Face

**📢 New Release:** We’ve released [ granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M), the successor to **SmolDocling**. It will now receive updates and support, check it out! 

![SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview/resolve/main/assets/SmolDocling_doctags1.png)

### SmolDocling-256M-preview

SmolDocling is a multimodal Image-Text-to-Text model designed for efficient document conversion. It retains Docling's most popular features while ensuring full compatibility with Docling through seamless support for **DoclingDocuments**.

This model was presented in the paper [SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion](https://huggingface.co/papers/2503.11576).

###  🚀 Features: 

- 🏷️ **DocTags for Efficient Tokenization** – Introduces DocTags an efficient and minimal representation for documents that is fully compatible with **DoclingDocuments**. 
- 🔍 **OCR (Optical Character Recognition)** – Extracts text accurately from images. 
- 📐 **Layout and Localization** – Preserves document structure and document element **bounding boxes**. 
- 💻 **Code Recognition** – Detects and formats code blocks including identation. 
- 🔢 **Formula Recognition** – Identifies and processes mathematical expressions. 
- 📊 **Chart Recognition** – Extracts and interprets chart data. 
- 📑 **Table Recognition** – Supports column and row headers for structured table extraction. 
- 🖼️ **Figure Classification** – Differentiates figures and graphical elements. 
- 📝 **Caption Correspondence** – Links captions to relevant images and figures. 
- 📜 **List Grouping** – Organizes and structures list elements correctly. 
- 📄 **Full-Page Conversion** – Processes entire pages for comprehensive document conversion including all page elements (code, equations, tables, charts etc.) 
- 🔲 **OCR with Bounding Boxes** – OCR regions using a bounding box.
- 📂 **General Document Processing** – Trained for both scientific and non-scientific documents. 
- 🔄 **Seamless Docling Integration** – Import into **Docling** and export in multiple formats.
- 💨 **Fast inference using VLLM** – Avg of 0.35 secs per page on A100 GPU.

###   🚧 *Coming soon!*  

- 📊 **Better chart recognition 🛠️**
- 📚 **One shot multi-page inference ⏱️**
- 🧪 **Chemical Recognition**
- 📙 **Datasets**

##  ⌨️ Get started (code examples) 

You can use **transformers**, **vllm**, or **onnx** to perform inference, and [Docling](https://github.com/docling-project/docling) to convert results to variety of output formats (md, html, etc.):

> [!NOTE]- 📄 Single page image inference using Tranformers 🤖
> ```python
> # Prerequisites:
> # pip install torch
> # pip install docling_core
> # pip install transformers
> 
> import torch
> from docling_core.types.doc import DoclingDocument
> from docling_core.types.doc.document import DocTagsDocument
> from transformers import AutoProcessor, AutoModelForVision2Seq
> from transformers.image_utils import load_image
> from pathlib import Path
> 
> DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
> 
> # Load images
> image = load_image("https://upload.wikimedia.org/wikipedia/commons/7/76/GazettedeFrance.jpg")
> 
> # Initialize processor and model
> processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
> model = AutoModelForVision2Seq.from_pretrained(
>     "ds4sd/SmolDocling-256M-preview",
>     torch_dtype=torch.bfloat16,
>     _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
> ).to(DEVICE)
> 
> # Create input messages
> messages = [
>     {
>         "role": "user",
>         "content": [
>             {"type": "image"},
>             {"type": "text", "text": "Convert this page to docling."}
>         ]
>     },
> ]
> 
> # Prepare inputs
> prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
> inputs = processor(text=prompt, images=[image], return_tensors="pt")
> inputs = inputs.to(DEVICE)
> 
> # Generate outputs
> generated_ids = model.generate(**inputs, max_new_tokens=8192)
> prompt_length = inputs.input_ids.shape[1]
> trimmed_generated_ids = generated_ids[:, prompt_length:]
> doctags = processor.batch_decode(
>     trimmed_generated_ids,
>     skip_special_tokens=False,
> )[0].lstrip()
> 
> # Populate document
> doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
> print(doctags)
> # create a docling document
> doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
> 
> # export as any format
> # HTML
> # Path("Out/").mkdir(parents=True, exist_ok=True)
> # output_path_html = Path("Out/") / "example.html"
> # doc.save_as_html(output_path_html)
> # MD
> print(doc.export_to_markdown())
> ```

> [!NOTE]- 🚀 Fast Batch Inference Using VLLM
> ```python
> # Prerequisites:
> # pip install vllm
> # pip install docling_core
> # place page images you want to convert into "img/" dir
> 
> import time
> import os
> from vllm import LLM, SamplingParams
> from PIL import Image
> from docling_core.types.doc import DoclingDocument
> from docling_core.types.doc.document import DocTagsDocument
> from pathlib import Path
> 
> # Configuration
> MODEL_PATH = "ds4sd/SmolDocling-256M-preview"
> IMAGE_DIR = "img/"  # Place your page images here
> OUTPUT_DIR = "out/"
> PROMPT_TEXT = "Convert page to Docling."
> 
> # Ensure output directory exists
> os.makedirs(OUTPUT_DIR, exist_ok=True)
> 
> # Initialize LLM
> llm = LLM(model=MODEL_PATH, limit_mm_per_prompt={"image": 1})
> 
> sampling_params = SamplingParams(
>     temperature=0.0,
>     max_tokens=8192)
> 
> chat_template = f"<|im_start|>User:<image>{PROMPT_TEXT}<end_of_utterance>
> Assistant:"
> 
> image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
> 
> start_time = time.time()
> total_tokens = 0
> 
> for idx, img_file in enumerate(image_files, 1):
>     img_path = os.path.join(IMAGE_DIR, img_file)
>     image = Image.open(img_path).convert("RGB")
> 
>     llm_input = {"prompt": chat_template, "multi_modal_data": {"image": image}}
>     output = llm.generate([llm_input], sampling_params=sampling_params)[0]
>     
>     doctags = output.outputs[0].text
>     img_fn = os.path.splitext(img_file)[0]
>     output_filename = img_fn + ".dt"
>     output_path = os.path.join(OUTPUT_DIR, output_filename)
> 
>     with open(output_path, "w", encoding="utf-8") as f:
>         f.write(doctags)
> 
>     # To convert to Docling Document, MD, HTML, etc.:
>     doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
>     doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
>     # export as any format
>     # HTML
>     # output_path_html = Path(OUTPUT_DIR) / f"{img_fn}.html"
>     # doc.save_as_html(output_path_html)
>     # MD
>     output_path_md = Path(OUTPUT_DIR) / f"{img_fn}.md"
>     doc.save_as_markdown(output_path_md)
> print(f"Total time: {time.time() - start_time:.2f} sec")
> ```

> [!NOTE]- ONNX Inference
> ```python
> # Prerequisites:
> # pip install onnxruntime
> # pip install onnxruntime-gpu
> from transformers import AutoConfig, AutoProcessor
> from transformers.image_utils import load_image
> import onnxruntime
> import numpy as np
> import os
> from docling_core.types.doc import DoclingDocument
> from docling_core.types.doc.document import DocTagsDocument
> 
> os.environ["OMP_NUM_THREADS"] = "1"
> # cuda
> os.environ["ORT_CUDA_USE_MAX_WORKSPACE"] = "1"
> 
> # 1. Load models
> ## Load config and processor
> model_id = "ds4sd/SmolDocling-256M-preview"
> config = AutoConfig.from_pretrained(model_id)
> processor = AutoProcessor.from_pretrained(model_id)
> 
> ## Load sessions
> # !wget https://huggingface.co/ds4sd/SmolDocling-256M-preview/resolve/main/onnx/vision_encoder.onnx
> # !wget https://huggingface.co/ds4sd/SmolDocling-256M-preview/resolve/main/onnx/embed_tokens.onnx
> # !wget https://huggingface.co/ds4sd/SmolDocling-256M-preview/resolve/main/onnx/decoder_model_merged.onnx
> # cpu
> # vision_session = onnxruntime.InferenceSession("vision_encoder.onnx")
> # embed_session = onnxruntime.InferenceSession("embed_tokens.onnx")
> # decoder_session = onnxruntime.InferenceSession("decoder_model_merged.onnx"
> 
> # cuda
> vision_session = onnxruntime.InferenceSession("vision_encoder.onnx", providers=["CUDAExecutionProvider"])
> embed_session = onnxruntime.InferenceSession("embed_tokens.onnx", providers=["CUDAExecutionProvider"])
> decoder_session = onnxruntime.InferenceSession("decoder_model_merged.onnx", providers=["CUDAExecutionProvider"])
> 
> ## Set config values
> num_key_value_heads = config.text_config.num_key_value_heads
> head_dim = config.text_config.head_dim
> num_hidden_layers = config.text_config.num_hidden_layers
> eos_token_id = config.text_config.eos_token_id
> image_token_id = config.image_token_id
> end_of_utterance_id = processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
> 
> # 2. Prepare inputs
> ## Create input messages
> messages = [
>     {
>         "role": "user",
>         "content": [
>             {"type": "image"},
>             {"type": "text", "text": "Convert this page to docling."}
>         ]
>     },
> ]
> 
> ## Load image and apply processor
> image = load_image("https://ibm.biz/docling-page-with-table")
> prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
> inputs = processor(text=prompt, images=[image], return_tensors="np")
> 
> ## Prepare decoder inputs
> batch_size = inputs['input_ids'].shape[0]
> past_key_values = {
>     f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
>     for layer in range(num_hidden_layers)
>     for kv in ('key', 'value')
> }
> image_features = None
> input_ids = inputs['input_ids']
> attention_mask = inputs['attention_mask']
> position_ids = np.cumsum(inputs['attention_mask'], axis=-1)
> 
> 
> # 3. Generation loop
> max_new_tokens = 8192
> generated_tokens = np.array([[]], dtype=np.int64)
> for i in range(max_new_tokens):
>   inputs_embeds = embed_session.run(None, {'input_ids': input_ids})[0]
> 
>   if image_features is None:
>     ## Only compute vision features if not already computed
>     image_features = vision_session.run(
>         ['image_features'],  # List of output names or indices
>         {
>             'pixel_values': inputs['pixel_values'],
>             'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
>         }
>     )[0]
>     
>     ## Merge text and vision embeddings
>     inputs_embeds[inputs['input_ids'] == image_token_id] = image_features.reshape(-1, image_features.shape[-1])
> 
>   logits, *present_key_values = decoder_session.run(None, dict(
>       inputs_embeds=inputs_embeds,
>       attention_mask=attention_mask,
>       position_ids=position_ids,
>       **past_key_values,
>   ))
> 
>   ## Update values for next generation loop
>   input_ids = logits[:, -1].argmax(-1, keepdims=True)
>   attention_mask = np.ones_like(input_ids)
>   position_ids = position_ids[:, -1:] + 1
>   for j, key in enumerate(past_key_values):
>     past_key_values[key] = present_key_values[j]
> 
>   generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
>   if (input_ids == eos_token_id).all() or (input_ids == end_of_utterance_id).all():
>     break  # Stop predicting
> 
> doctags = processor.batch_decode(
>     generated_tokens,
>     skip_special_tokens=False,
> )[0].lstrip()
> 
> print(doctags)
> 
> doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
> print(doctags)
> # create a docling document
> doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
> 
> print(doc.export_to_markdown())
> ```

💻 Local inference on Apple Silicon with MLX: [see here](https://huggingface.co/ds4sd/SmolDocling-256M-preview-mlx-bf16)

##  DocTags 

![Image description](https://huggingface.co/ds4sd/SmolDocling-256M-preview/resolve/main/assets/doctags_v2.png){width=800 height=auto}

##  Supported Instructions 

Description

Instruction

Comment

Full conversion

Chart

Formula

Code

Table

[Lysak et al., 2023](https://arxiv.org/pdf/2305.03393)

Actions and Pipelines

####  📊 Datasets 

- [SynthCodeNet](https://huggingface.co/datasets/ds4sd/SynthCodeNet)
- [SynthFormulaNet](https://huggingface.co/datasets/ds4sd/SynthFormulaNet)
- [SynthChartNet](https://huggingface.co/datasets/ds4sd/SynthChartNet)
- [DoclingMatix](https://huggingface.co/datasets/HuggingFaceM4/DoclingMatix)

####  Model Summary 

- **Developed by:** Docling Team, IBM Research
- **Model type:** Multi-modal model (image\+text)
- **Language(s) (NLP):** English
- **License:** Apache 2.0
- **Architecture:** Based on [Idefics3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3) (see technical summary)
- **Finetuned from model:** Based on [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)

**Repository:** [Docling](https://github.com/docling-project/docling)

**Paper:** [arXiv](https://arxiv.org/abs/2503.11576)

**Project Page:** [Hugging Face](https://huggingface.co/ds4sd/SmolDocling-256M-preview)

**Citation:**

```
@misc{nassar2025smoldoclingultracompactvisionlanguagemodel,
      title={SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion}, 
      author={Ahmed Nassar and Andres Marafioti and Matteo Omenetti and Maksym Lysak and Nikolaos Livathinos and Christoph Auer and Lucas Morin and Rafael Teixeira de Lima and Yusik Kim and A. Said Gurbuz and Michele Dolfi and Miquel Farré and Peter W. J. Staar},
      year={2025},
      eprint={2503.11576},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.11576}, 
}
```

**Demo:** [HF Space](https://huggingface.co/spaces/ds4sd/SmolDocling-256M-Demo)
