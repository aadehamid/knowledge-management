title: baidu/Unlimited-OCR — 3B · DENSE · 32K ctx
description: Baidu's state-of-the-art document-parsing model with Reference Sliding Window Attention \(R-SWA\), optimized for full-page OCR and markdown generation.

# baidu/Unlimited-OCR — 3B · DENSE · 32K ctx

## Overview {#overview}

[Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) is Baidu's DeepSeek-OCR-lineage document-parsing model. It shares the DeepSeek-OCR *gundam* vision stack (SAM-ViT-B \+ CLIP-L DeepEncoder, `base_size=1024` / `image_size=640` / crop) and ships with the same n-gram logits processor for high-quality, repetition-free document → markdown generation.

## Prerequisites {#prerequisites}

- **Hardware**: a single GPU with >\=8 GB VRAM is enough for BF16 inference.
- **vLLM**: served from the dedicated release image (the architecture is not yet in a stable pip wheel). Pull the image shown in the Install → Docker tab.

## Serving recipe (required) {#serving-recipe-required}

The model has **no chat template** and is trained for a specific prompt/decode recipe — without it the model returns empty output. The required pieces:

- register the no-repeat-ngram logits processor on the server: `--logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor`
- prompt text must begin with a literal `<image>` (e.g. `<image>document parsing.`)
- `skip_special_tokens=False`
- pass the processor args per request: `ngram_size=35`, `window_size=128` (use `window_size=1024` for multi-page / PDF input)

## Client Usage {#client-usage}

### Online OCR serving {#online-ocr-serving}

```bash
docker run --rm --gpus all --network host --ipc host \
  vllm/vllm-openai:unlimited-ocr \
  baidu/Unlimited-OCR \
  --trust-remote-code \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
```

```python
import time
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=3600)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "<image>document parsing."},
            {"type": "image_url", "image_url": {"url": "https://huggingface.co/baidu/Unlimited-OCR/resolve/main/assets/baidu.png"}},
        ],
    }
]

start = time.time()
response = client.chat.completions.create(
    model="baidu/Unlimited-OCR",
    messages=messages,
    max_tokens=8192,
    temperature=0.0,
    extra_body={
        "skip_special_tokens": False,
        "vllm_xargs": {"ngram_size": 35, "window_size": 128},
    },
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")
```

The raw output carries `<|ref|>…<|/ref|>` / `<|det|>…<|/det|>` grounding tokens; unwrap the `<|ref|>` text and drop the `<|det|>` coordinate boxes to get clean markdown.

## Troubleshooting / Configuration Tips {#troubleshooting--configuration-tips}

- **The custom logits processor is required** — without it long documents loop on `<|det|>` coordinate tokens.
- Empty output almost always means the prompt is missing the literal `<image>` prefix or `skip_special_tokens` was left `True`.
- OCR tasks don't benefit from prefix caching / image reuse, so prefix caching and the mm-processor cache are disabled in the recipe.
- Single-image input uses gundam (crop) mode; multi-image requests automatically fall back to non-crop (base) mode — use `window_size=1024` for those.

## References {#references}

- [Unlimited-OCR on Hugging Face](https://huggingface.co/baidu/Unlimited-OCR)
- [vLLM multimodal inputs guide](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html#offline-inference)
