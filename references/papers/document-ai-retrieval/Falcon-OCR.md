title: tiiuae/Falcon-OCR · Hugging Face
description: We’re on a journey to advance and democratize artificial intelligence through open source and open science.

![Falcon OCR Logo](https://cdn-uploads.huggingface.co/production/uploads/663c9939c1b4f7297c4ae6f6/YIuxzgDiV5T2ZuSB4bam9.png)

#  Falcon OCR 

Falcon OCR is a 300M parameter early-fusion vision-language model for document OCR. Given an image, it can produce plain text, LaTeX for formulas, or HTML for tables, depending on the requested output format.

Most OCR VLM systems are built as a pipeline with a vision encoder feeding a separate text decoder, plus additional task-specific glue. Falcon OCR takes a different approach: a single Transformer processes image patches and text tokens in a shared parameter space from the first layer, using a hybrid attention mask where image tokens attend bidirectionally and text tokens decode causally conditioned on the image.

We built it this way for two practical reasons. First, it keeps the interface simple: one backbone, one decoding path, and task switching through prompts rather than a growing set of modules. Second, a 0.3B model has a lower latency and cost footprint than 0.9B-class OCR VLMs, and in our vLLM-based serving setup this translates into higher throughput, often 2–3× faster depending on sequence lengths and batch configuration. To our knowledge, this is one of the first attempts to apply this early-fusion single-stack recipe directly to competitive document OCR at this scale.

###  Links 

- Code and inference engine: [https://github.com/tiiuae/Falcon-Perception](https://github.com/tiiuae/Falcon-Perception)
- Tech report: [https://arxiv.org/pdf/2603.27365](https://arxiv.org/pdf/2603.27365)
- Perception model: `tiiuae/falcon-perception`
- vLLM/Docker: [https://ghcr.io/tiiuae/falcon-ocr:latest](https://ghcr.io/tiiuae/falcon-ocr:latest)

##  Quickstart 

###  Installation 

```bash
pip install "torch>=2.5" transformers pillow einops
```

Falcon OCR requires PyTorch 2.5 or newer for FlexAttention. The first call may be slower as `torch.compile` builds optimized kernels.

###  Single-Image OCR 

```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/Falcon-OCR",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
image = Image.open("document.png")
texts = model.generate(image)  # default category is "plain"
print(texts[0])
```

###   Choose an output format with `category`  

```python
texts = model.generate(image, category="text")     # plain text
texts = model.generate(image, category="formula")  # LaTeX
texts = model.generate(image, category="table")    # HTML table
```

##  API 

###   `model.generate(images, category="plain", **kwargs)`  

- **Inputs**:
    - `images`: a `PIL.Image.Image` or a list of images
    - `category`: one of `plain`, `text`, `table`, `formula`, `caption`, `footnote`, `list-item`, `page-footer`, `page-header`, `section-header`, `title`
- **Returns**: `list[str]`, one extracted string per image

##  Layout OCR (Two-Stage Pipeline) 

For sparse documents, running OCR on the whole image can work well. For dense documents with heterogeneous regions (multi-column layouts, interleaved tables and formulas, small captions), we provide an optional two-stage pipeline:

1. A layout detector finds regions on the page.
2. Falcon OCR runs independently on each crop with a category-specific prompt. We use [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) as the layout detector.

```python
results = model.generate_with_layout(image)
for det in results[0]:
    print(f"[{det['category']}] {det['text'][:100]}...")
```

Batch mode:

```python
results = model.generate_with_layout(
    [Image.open("page1.png"), Image.open("page2.png")],
    ocr_batch_size=32,
)
```

The layout model is loaded lazily on the first `generate_with_layout()` call and runs on the same GPU as the OCR model. **Returns**: `list[list[dict]]`, one list per image, in reading order:

```python
{
    "category": "text",       # layout category
    "bbox": [x1, y1, x2, y2], # in original image pixels
    "score": 0.93,            # detection confidence
    "text": "..."             # extracted text
}
```

##  When to Use What 

| Mode | Best for | How |
|----|----|----|
| **Plain OCR** | Simple documents, real-world photos, slides, receipts, invoices | `model.generate(image)` |
| **Layout \+ OCR** | Complex multi-column documents, academic papers, reports, dense pages like newspapers | `model.generate_with_layout(image)` |

##  Benchmark Results 

> [!NOTE]+ olmOCR Benchmark
> Category-wise performance comparison of FalconOCR against state-of-the-art OCR models. We report accuracy (%) across all category splits.
> | Model | Average | ArXiv Math | Base | Hdr/Ftr | TinyTxt | MultCol | OldScan | OldMath | Tables |
> |----|----|----|----|----|----|----|----|----|----|
> | Mistral OCR 3 | 81.7 | **85.4** | **99.9** | 93.8 | 88.9 | 82.1 | 48.8 | 68.3 | 86.1 |
> | Chandra | **82.0** | 81.4 | 99.8 | 88.8 | **91.9** | 82.9 | **49.2** | 73.6 | 88.2 |
> | Gemini 3 Pro | 80.2 | 70.6 | 99.8 | 84.0 | 90.3 | 79.2 | 47.5 | 84.9 | 84.9 |
> | PaddleOCR VL 1.5 | 79.3 | **85.4** | 98.8 | **96.9** | 80.8 | 82.6 | 39.2 | 66.4 | 84.1 |
> | PaddleOCR VL | 79.2 | **85.4** | 98.6 | **96.9** | 80.8 | 82.5 | 38.8 | 66.4 | 83.9 |
> | DeepSeek OCR v2 | 78.8 | 81.9 | 99.8 | 95.6 | 88.7 | 83.6 | 33.7 | 68.8 | 78.1 |
> | Gemini 3 Flash | 77.5 | 66.5 | 99.8 | 83.8 | 88.2 | 73.7 | 46.0 | **85.8** | 75.9 |
> | GPT 5.2 | 69.8 | 61.0 | 99.8 | 75.6 | 62.2 | 70.2 | 34.6 | 75.8 | 79.0 |
> | **FalconOCR** | 80.3 | 80.5 | 99.5 | 94.0 | 78.5 | **87.1** | 43.5 | 69.2 | **90.3** |

> [!NOTE]- OmniDocBench
> Performance comparison on full-page document parsing. Overall↑ aggregates the three sub-metrics. Edit↓ measures text edit distance (lower is better). CDM↑ evaluates formula recognition accuracy. TEDS↑ measures table structure similarity.
> | Model | Overall↑ | Edit↓ | CDM↑ | TEDS↑ |
> |----|----|----|----|----|
> | PaddleOCR VL 1.5 | **94.37** | 0.025 | **94.4** | **91.1** |
> | PaddleOCR VL | 91.76 | **0.024** | 91.7 | 85.9 |
> | Chandra | 88.97 | 0.046 | 88.1 | 89.5 |
> | DeepSeek OCR v2 | 87.66 | 0.037 | 89.2 | 77.5 |
> | GPT 5.2 | 86.56 | 0.061 | 88.0 | 77.7 |
> | Mistral OCR 3 | 85.20 | 0.053 | 84.3 | 76.1 |
> | **FalconOCR** | 88.64 | 0.055 | 86.8 | 84.6 |

###  Results Analysis 

First, a compact model can be competitive when the interface is simple and the training signal is targeted. On olmOCR, Falcon OCR performs strongly on multi-column documents and tables, and is competitive overall against substantially larger systems. Second, evaluation on full-page parsing is sensitive to matching and representation details. On OmniDocBench, the table and formula metrics depend not only on recognition quality but also on how predicted elements are matched to ground truth and how output structure is normalized.

More broadly, these results suggest that an early-fusion single-stack Transformer can be a viable alternative to the common "vision encoder plus text decoder" recipe for OCR. We do not view this as a finished answer, but as a promising direction: one early-fusion backbone, a shared parameter space between text and images, a single decoding interface, and better data and training signals, rather than increasingly complex pipelines. To our knowledge, this is among the first demonstrations that this early-fusion recipe can reach competitive document OCR accuracy at this scale, and we hope it encourages further work in this direction.

##  Serving Throughput 

Measured on a single A100-80GB GPU with vLLM, processing document images from olmOCR-Bench under high concurrency for optimal vLLM utilization.

- **Layout \+ OCR** — The full end-to-end pipeline: layout detection finds regions on each page, crops them, and vLLM runs OCR on every crop. This represents the real-world serving throughput, inclusive of both layout detection and OCR time.

| Mode | tok/s | img/s | Description |
|----|---:|---:|----|
| **Layout \+ OCR** | 5,825 | 2.9 | Full pipeline: layout detection → crop → per-region OCR |

At 0.3B parameters, Falcon OCR is roughly 3× smaller than 0.9B-class OCR VLMs (e.g., PaddleOCR VL), which translates directly into higher serving throughput at competitive accuracy.

##  Limitations 

- **Old scans and tiny text**: Heavily degraded scans and very small glyphs remain challenging. These cases often require higher effective resolution and better coverage in the training mixture.
- **Non-unique table representations**: Visually identical tables can be encoded in structurally different HTML forms, which can affect tree-based metrics.
- **Formula matching sensitivity**: LaTeX and Unicode conventions can be penalized differently depending on the benchmark normalization and matching pipeline.

##  Examples 

*Click each section below to expand.*

> [!NOTE]+ Handwriting and Real World Images
>  ![Handwriting and real world OCR examples](https://cdn-uploads.huggingface.co/production/uploads/62fe441427c98b09b503a4e3/51Fj1wxxtAV_jwubml6sa.png){width=600} 

> [!NOTE]- Tables
>  ![Table OCR examples](https://cdn-uploads.huggingface.co/production/uploads/62fe441427c98b09b503a4e3/2yZjZJAEHVVpd_jfyyDcQ.png){width=600} 

> [!NOTE]- Formulas
>  ![Formula OCR examples](https://cdn-uploads.huggingface.co/production/uploads/62fe441427c98b09b503a4e3/__XMb0GyGO02IPKlQsPQx.png){width=600} 

> [!NOTE]- Complex Layout
>  ![Complex layout OCR examples](https://cdn-uploads.huggingface.co/production/uploads/62fe441427c98b09b503a4e3/kTR7nI7ogEqdI1SQtXpTu.png){width=600} 

---

##  vLLM Server 

We also provide a Docker-based vLLM-backed inference server capable of serving approximately 6,000 tokens per second.

Single Docker image with two services:

| Service | Default Port | Description |
|----|----|----|
| **vLLM** | 8000 | Falcon-OCR vision-language model (OpenAI-compatible API) |
| **Pipeline** | 5002 | Full document parsing: layout detection → crop → OCR → markdown |

The layout model runs inside the pipeline process — it is not a standalone service.

###  Quick Start 

```bash
docker run -d --name falcon-ocr \
  --gpus '"device=0,1"' \
  -e EXPOSED_GPU_IDS=0,1 \
  -e VLLM_GPU=0 \
  -e PIPELINE_GPU=1 \
  -e VLLM_GPU_MEM_UTIL=0.90 \
  -p 8000:8000 \
  -p 5002:5002 \
  ghcr.io/tiiuae/falcon-ocr:latest
```

###  API 

> [!NOTE]+ Health Checks
> ```bash
> curl http://localhost:8000/health      # vLLM
> curl http://localhost:5002/health      # Pipeline
> ```

> [!NOTE]- Upload (multipart file upload — images and PDFs)
> The easiest way to send files. Supports images and multi-page PDFs:
> ```bash
> # Single image
> curl -X POST http://localhost:5002/falconocr/upload \
>   -F "files=@photo.jpg;type=image/jpeg"
> # PDF document
> curl -X POST http://localhost:5002/falconocr/upload \
>   -F "files=@document.pdf;type=application/pdf"
> ```

> [!NOTE]- Parse (full pipeline: layout + OCR)
> Send base64-encoded images for layout detection, cropping, and OCR:
> ```bash
> curl -X POST http://localhost:5002/falconocr/parse \
>   -H "Content-Type: application/json" \
>   -d '{
>     "images": ["data:image/jpeg;base64,<...>"],
>     "skip_layout": false
>   }'
> ```
> Response:
> ```json
> {
>   "json_result": [[{
>     "index": 0,
>     "mapped_label": "text",
>     "content": "The Manuscript",
>     "bbox": [273, 273, 937, 380],
>     "score": 0.3145
>   }]],
>   "markdown_result": "The Manuscript",
>   "total_output_tokens": 93,
>   "processing_time_ms": 414
> }
> ```

> [!NOTE]- Parse (direct VLM, no layout)
> Skip layout detection and send the full image directly to the VLM:
> ```bash
> curl -X POST http://localhost:5002/falconocr/parse \
>   -H "Content-Type: application/json" \
>   -d '{
>     "images": ["data:image/jpeg;base64,<...>"],
>     "skip_layout": true
>   }'
> ```

> [!NOTE]- Direct vLLM (OpenAI-compatible)
> ```bash
> curl -X POST http://localhost:8000/v1/chat/completions \
>   -H "Content-Type: application/json" \
>   -d '{
>     "model": "falcon-ocr",
>     "messages": [{"role": "user", "content": [
>       {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}},
>       {"type": "text", "text": "Extract the text content from this image.\n<|OCR_PLAIN|>"}
>     ]}],
>     "max_tokens": 2048
>   }'
> ```

###  Configuration 

All settings are controlled via environment variables at `docker run` time.

> [!NOTE]+ GPU Assignment
> | Variable | Default | Description |
> |----|----|----|
> | `VLLM_GPU` | `0` | Host GPU ID for the vLLM process |
> | `PIPELINE_GPU` | `0` | Host GPU ID for the pipeline (layout model) |
> | `EXPOSED_GPU_IDS` | *(all visible)* | Comma-separated host GPU IDs passed via `--gpus` (for index remapping) |

> [!NOTE]- Port Assignment
> | Variable | Default | Description |
> |----|----|----|
> | `VLLM_PORT` | `8000` | Port for the vLLM OpenAI-compatible API |
> | `PIPELINE_PORT` | `5002` | Port for the pipeline API |

> [!NOTE]- vLLM Tuning
> | Variable | Default | Description |
> |----|----|----|
> | `VLLM_GPU_MEM_UTIL` | `0.90` | Fraction of GPU memory vLLM can use |
> | `MAX_NUM_SEQS` | `2048` | Max concurrent sequences in vLLM |
> | `MAX_MODEL_LEN` | `8192` | Max model context length |
> | `DTYPE` | `bfloat16` | Model dtype |
> | `MAX_NUM_BATCHED_TOKENS` | *(auto)* | Max batched tokens per iteration |
> | `CHUNKED_PREFILL` | `false` | Enable chunked prefill |

> [!NOTE]- Layout Model Tuning
> | Variable | Default | Description |
> |----|----|----|
> | `LAYOUT_BATCH_SIZE` | `64` | Batch size for layout detection inference |

> [!NOTE]- Model Paths
> | Variable | Default | Description |
> |----|----|----|
> | `FALCON_OCR_MODEL` | `/models/Falcon-OCR` | Path to Falcon-OCR VLM weights (inside container) |
> | `SERVED_MODEL_NAME` | `falcon-ocr` | Model name exposed by vLLM API |

###  Deployment Modes 

> [!NOTE]+ Two GPUs (best throughput)
> vLLM on one GPU, layout model on another — zero GPU contention:
> ```bash
> docker run -d --name falcon-ocr \
>   --gpus '"device=3,4"' \
>   -e EXPOSED_GPU_IDS=3,4 \
>   -e VLLM_GPU=3 \
>   -e PIPELINE_GPU=4 \
>   -e VLLM_GPU_MEM_UTIL=0.90 \
>   -p 8000:8000 \
>   -p 5002:5002 \
>   ghcr.io/tiiuae/falcon-ocr:latest
> ```

> [!NOTE]- Single GPU (memory sharing)
> Both services share one GPU — tune `VLLM_GPU_MEM_UTIL` to leave room for the layout model:
> ```bash
> docker run -d --name falcon-ocr \
>   --gpus '"device=0"' \
>   -e EXPOSED_GPU_IDS=0 \
>   -e VLLM_GPU=0 \
>   -e PIPELINE_GPU=0 \
>   -e VLLM_GPU_MEM_UTIL=0.55 \
>   -e LAYOUT_BATCH_SIZE=32 \
>   -e MAX_NUM_SEQS=512 \
>   -p 8000:8000 \
>   -p 5002:5002 \
>   ghcr.io/tiiuae/falcon-ocr:latest
> ```

> [!NOTE]- Custom Ports
> ```bash
> docker run -d --name falcon-ocr \
>   --gpus '"device=0,1"' \
>   -e EXPOSED_GPU_IDS=0,1 \
>   -e VLLM_GPU=0 \
>   -e PIPELINE_GPU=1 \
>   -e VLLM_PORT=18000 \
>   -e PIPELINE_PORT=15002 \
>   -p 18000:18000 \
>   -p 15002:15002 \
>   ghcr.io/tiiuae/falcon-ocr:latest
> ```
> Docker `--gpus "device=3,4"` makes the container see GPUs as local indices `0,1`. `EXPOSED_GPU_IDS=3,4` allows you to reference host GPU IDs (`VLLM_GPU=3`, `PIPELINE_GPU=4`); the entrypoint remaps them to the correct container-local indices.

##  Citation 

If you use Falcon OCR, please cite:

```bibtex
@article{bevli2026falcon,
  title   = {Falcon Perception},
  author  = {Bevli, Aviraj and Chaybouti, Sofian and Dahou, Yasser and Hacid, Hakim and Huynh, Ngoc Dung and Le Khac, Phuc H. and Narayan, Sanath and Para, Wamiq Reyaz and Singh, Ankit},
  journal = {arXiv preprint arXiv:2603.27365},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.27365}
}
```
