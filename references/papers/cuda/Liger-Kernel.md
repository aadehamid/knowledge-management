title: GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training
description: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.

# GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training

StableNightlyDiscord

[ ![Downloads (Stable)](https://camo.githubusercontent.com/38f28dab3ba8841b4d66af4c6945ea7f2379b9d056c48b4f99641ad0f4342c09/68747470733a2f2f7374617469632e706570792e746563682f62616467652f6c696765722d6b65726e656c) ](https://pepy.tech/project/liger-kernel)

[ ![PyPI - Version](https://camo.githubusercontent.com/783a02a2cd20d43408adbaeca48dfdc2f85ba64ea05302e97fd3673042f55d8d/68747470733a2f2f696d672e736869656c64732e696f2f707970692f762f6c696765722d6b65726e656c3f636f6c6f723d677265656e) ](https://pypi.org/project/liger-kernel)

[ ![Downloads (Nightly)](https://camo.githubusercontent.com/cd172da3aa68faa03a7b196bd4628f65a47918b456d729bc9b7cb489f53f4b91/68747470733a2f2f7374617469632e706570792e746563682f62616467652f6c696765722d6b65726e656c2d6e696768746c79) ](https://pepy.tech/project/liger-kernel-nightly)

[ ![PyPI - Version](https://camo.githubusercontent.com/ae97aa19ab1230846c59bf0438564251d1b4c5e1f70bc2130296ec31e4015413/68747470733a2f2f696d672e736869656c64732e696f2f707970692f762f6c696765722d6b65726e656c2d6e696768746c793f636f6c6f723d677265656e) ](https://pypi.org/project/liger-kernel-nightly)

[ ![Join Our Discord](https://camo.githubusercontent.com/92a50f77b3f2515ed63c67b24661868139973908b69d250d1c7e75cc1ca464f3/68747470733a2f2f646362616467652e6c696d65732e70696e6b2f6170692f7365727665722f68747470733a2f2f646973636f72642e67672f58344d61785067413f7374796c653d666c6174) ](https://discord.gg/X4MaxPgA)

[![](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/logo-banner.png)](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/logo-banner.png)

[Installation](#installation) | [Getting Started](#getting-started) | [Examples](#examples) | [High-level APIs](#high-level-apis) | [Low-level APIs](#low-level-apis) | [Cite our work](#cite-this-work)

> [!NOTE]- Latest News 🔥
> - \[2025/12/19\] We announced a liger kernel discord channel at [https://discord.gg/X4MaxPgA](https://discord.gg/X4MaxPgA); We will be hosting Liger Kernel x Triton China Meetup in mid of January 2026
> - \[2025/03/06\] We release a joint blog post on TorchTune × Liger - [Peak Performance, Minimized Memory: Optimizing torchtune’s performance with torch.compile & Liger Kernel](https://pytorch.org/blog/peak-performance-minimized-memory/)
> - \[2024/12/11\] We release [v0.5.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.5.0): 80% more memory efficient post training losses (DPO, ORPO, CPO, etc)!
> - \[2024/12/5\] We release LinkedIn Engineering Blog - [Liger-Kernel: Empowering an open source ecosystem of Triton Kernels for Efficient LLM Training](https://www.linkedin.com/blog/engineering/open-source/liger-kernel-open-source-ecosystem-for-efficient-llm-training)
> - \[2024/11/6\] We release [v0.4.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.4.0): Full AMD support, Tech Report, Modal CI, Llama-3.2-Vision!
> - \[2024/10/21\] We have released the tech report of Liger Kernel on Arxiv: [https://arxiv.org/pdf/2410.10989](https://arxiv.org/pdf/2410.10989)
> - \[2024/9/6\] We release v0.2.1 ([X post](https://x.com/liger_kernel/status/1832168197002510649)). 2500\+ Stars, 10\+ New Contributors, 50\+ PRs, 50k Downloads in two weeks!
> - \[2024/8/31\] CUDA MODE talk, [Liger-Kernel: Real-world Triton kernel for LLM Training](https://youtu.be/gWble4FreV4?si=dxPeIchhkJ36Mbns), [Slides](https://github.com/cuda-mode/lectures?tab=readme-ov-file#lecture-28-liger-kernel)
> - \[2024/8/23\] Official release: check out our [X post](https://x.com/hsu_byron/status/1827072737673982056)

**Liger Kernel** is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU **training throughput by 20%** and reduces **memory usage by 60%**. We have implemented **Hugging Face Compatible** `RMSNorm`, `RoPE`, `SwiGLU`, `CrossEntropy`, `FusedLinearCrossEntropy`, and more to come. The kernel works out of the box with [Flash Attention](https://github.com/Dao-AILab/flash-attention), [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), and [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed). We welcome contributions from the community to gather the best kernels for LLM training.

We've also added optimized Post-Training kernels that deliver **up to 80% memory savings** for alignment and distillation tasks. We support losses like DPO, CPO, ORPO, SimPO, KTO, JSD, and many more. Check out [how we optimize the memory](https://x.com/hsu_byron/status/1866577403918917655).

You can view the documentation site for additional installation, usage examples, and API references:[https://linkedin.github.io/Liger-Kernel/](https://linkedin.github.io/Liger-Kernel/)

You can view the Liger Kernel Technical Report: [https://openreview.net/forum?id\=36SjAIT42G](https://openreview.net/forum?id=36SjAIT42G)

[![Banner](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/banner.GIF)](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/banner.GIF)

With one line of code, Liger Kernel can increase throughput by more than 20% and reduce memory usage by 60%, thereby enabling longer context lengths, larger batch sizes, and massive vocabularies.

Speed UpMemory Reduction

[![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png)](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png)

[![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png)](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png)

> **Note:**
> - Benchmark conditions: LLaMA 3-8B, Batch Size \= 8, Data Type \= `bf16`, Optimizer \= AdamW, Gradient Checkpointing \= True, Distributed Strategy \= FSDP1 on 8 A100s.
> - Hugging Face models start to OOM at a 4K context length, whereas Hugging Face \+ Liger Kernel scales up to 16K.

 [![Post Training](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/post-training.png){width=50%}](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/post-training.png) 

We provide optimized post training kernels like DPO, ORPO, SimPO, and more which can reduce memory usage by up to 80%. You can easily use them as python modules.

```
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
orpo_loss = LigerFusedLinearORPOLoss()
y = orpo_loss(lm_head.weight, x, target)
```

| **Use Case** | **Description** |
|----|----|
| [**Hugging Face Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface) | Train LLaMA 3-8B \~20% faster with over 40% memory reduction on Alpaca dataset using 4 A100s with FSDP |
| [**Lightning Trainer**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/lightning) | Increase 15% throughput and reduce memory usage by 40% with LLaMA3-8B on MMLU dataset using 8 A100s with DeepSpeed ZeRO3 |
| [**Medusa Multi-head LLM (Retraining Phase)**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/medusa) | Reduce memory usage by 80% with 5 LM heads and improve throughput by 40% using 8 A100s with FSDP |
| [**Vision-Language Model SFT**](https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface/run_qwen2_vl.sh) | Finetune Qwen2-VL on image-text data using 4 A100s with FSDP |
| [**Liger ORPO Trainer**](https://github.com/linkedin/Liger-Kernel/blob/main/examples/alignment/run_orpo.py) | Align Llama 3.2 using Liger ORPO Trainer with FSDP with 50% memory reduction |

- **Ease of use:** Simply patch your Hugging Face model with one line of code, or compose your own model using our Liger Kernel modules.
- **Time and memory efficient:** In the same spirit as Flash-Attn, but for layers like **RMSNorm**, **RoPE**, **SwiGLU**, and **CrossEntropy**! Increases multi-GPU training throughput by 20% and reduces memory usage by 60% with **kernel fusion**, **in-place replacement**, and **chunking** techniques.
- **Exact:** Computation is exact—no approximations! Both forward and backward passes are implemented with rigorous unit tests and undergo convergence testing against training runs without Liger Kernel to ensure accuracy.
- **Lightweight:** Liger Kernel has minimal dependencies, requiring only Torch and Triton—no extra libraries needed! Say goodbye to dependency headaches!
- **Multi-GPU supported:** Compatible with multi-GPU setups (PyTorch FSDP, DeepSpeed, DDP, etc.).
- **Trainer Framework Integration**: [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory), [SFTTrainer](https://github.com/huggingface/trl/releases/tag/v0.10.1), [Hugging Face Trainer](https://github.com/huggingface/transformers/pull/32860), [SWIFT](https://github.com/modelscope/ms-swift), [oumi](https://github.com/oumi-ai/oumi/tree/main)

- `torch >= 2.1.2`
- `triton >= 2.3.0`

- `torch >= 2.5.0` Install according to the instruction in Pytorch official webpage.
- `triton >= 3.0.0` Install from pypi. (e.g. `pip install triton==3.0.0`)

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

- `torch == 2.9.0`
- `torch_npu == 2.9.0`
- `triton-ascend == 3.2.2` Install from the Ascend PyPI mirror (not on default PyPI).
- `CANN == 9.1.0`

```
pip install -e ".[dev]" --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
```

- `transformers >= 4.x`: Required if you plan to use the transformers models patching APIs. The specific model you are working will dictate the minimum version of transformers.
- `cuda-tile`: Required when enabling the optional cuTile backend on CUDA. Use this when your environment already provides CUDA Toolkit 13.1 or newer, or an existing tileiras compiler installation.
- `cuda-tile[tileiras]`: Required when enabling the optional cuTile backend with the tileiras compiler installed directly into your Python environment.
- `nvidia-cutlass-dsl >= 4.6.0`: Required when enabling the optional CuTe DSL backend on CUDA (the CUDA-only Python DSL shipped with NVIDIA CUTLASS, `import cutlass.cute`). Targets Hopper (SM90) and Blackwell (SM100/SM110).

> **Note:** Our kernels inherit the full spectrum of hardware compatibility offered by [Triton](https://github.com/triton-lang/triton).

To install the stable version:

```
$ pip install liger-kernel
```

To install the nightly version:

```
$ pip install liger-kernel-nightly
```

To install from source:

```
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel

# Install Default Dependencies
# Setup.py will detect the local backend and select default dependencies.
# On ROCm, install ROCm PyTorch first from the PyTorch ROCm index.
pip install -e .

# Setup Development Dependencies
pip install -e ".[dev]"

# ROCm source installs
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
# Then choose one:
pip install -e .
pip install -e ".[dev]"

# Setup cuTile Dependencies
pip install -e ".[cutile]"

# Or install cuTile with the optional tileiras compiler
pip install -e ".[cutile-tileiras]"

# Setup CuTe DSL (NVIDIA CUTLASS Python DSL) Dependencies
pip install -e ".[cutedsl]"
```

cuTile is an optional CUDA-only DSL implementation. After installing the `cutile` or `cutile-tileiras` extra, enable it explicitly:

```
LIGER_KERNEL_IMPL=cutile python your_script.py
```

`LIGER_KERNEL_IMPL` selects an opt-in implementation registered with Liger (currently `cutile` and `cutedsl`). Selecting one on an unsupported device, or without the required dependencies installed, raises an error.

CuTe DSL is the optional, CUDA-only Python DSL shipped with NVIDIA CUTLASS (`import cutlass.cute`), targeting Hopper (SM90) and Blackwell (SM100/SM110). After installing the `cutedsl` extra, enable it explicitly:

```
pip install "liger-kernel[cutedsl]"
LIGER_KERNEL_IMPL=cutedsl python your_script.py
```

It currently provides genuine `cutlass.cute` implementations of:

- **RMSNorm**
- **RoPE**
- **SwiGLU**
- **Cross entropy**
- **Fused linear cross entropy** (with an SM90-specialized variant)
- **Fused scaled cross entropy** (SM90)

Ops without a CuTe DSL kernel transparently fall back to the default Triton kernel.

The `cutedsl` extra also pulls in `apache-tvm-ffi`, which lets compiled kernels take PyTorch tensors directly rather than marshalling each one per call. It is optional — every kernel falls back to the marshalling launch without it — but short kernels are dominated by that per-call cost, so installing it is strongly recommended.

`LigerFusedLinearScaledCrossEntropyFunction` is an additional per-token operator, not a replacement for the reduction-oriented Triton `LigerFusedLinearCrossEntropyFunction`. It takes `input[M, H]`, `weight[V, H]`, and `target[M]`, applies `logits / temperature`, and returns FP32 negative log-likelihood `[M]` plus optional differentiable vocabulary entropy `[M]` in the input dtype. Reductions remain in PyTorch, and rows whose target equals `ignore_index` contribute zero outputs and gradients.

```
from liger_kernel.ops import LigerFusedLinearScaledCrossEntropyFunction

nll, entropy = LigerFusedLinearScaledCrossEntropyFunction.apply(
    x, weight, target, 1.0, -100, 1, True
)  # [M], [M]
loss = nll.sum() / (target != -100).sum().clamp_min(1)
```

The implementations share this public contract but use different schedules:

- **cuTile** (`LIGER_KERNEL_IMPL=cutile`) supports matching floating-point `input` and `weight` tensors on CUDA and a finite positive scalar `temperature`. The portable temporary-logits budget is 256 MiB. Large FP16/BF16 Blackwell workloads (`M >= 4096`, `V >= 131072`) automatically use 512 MiB to improve GEMM utilization. Set `LIGER_CUTILE_SCALED_CE_WORKSPACE_MB` to another positive MiB value for workload-specific tuning; for example, 1024 can help large combined NLL-plus-entropy workloads but is not universally faster. Backward reuses one workspace and writes `dX` and accumulates `dW` directly into their final tensors. `m_tiles_per_cluster` is accepted for API compatibility but does not change the cuTile schedule.
- **CuTe SM90** uses `LigerFusedScaledCrossEntropySM90Function` for BF16 inputs on Hopper. Its sole forward uses the fixed cluster-M2 N160 fragment kernel, with a measured split-N lookup for profiled long-sequence shapes, and never writes logits to HBM; `m_tiles_per_cluster` remains accepted for API compatibility but does not change that schedule. Backward runs `dZ`, `dX`, and `dW` in one persistent cluster kernel with a reusable 1024-token `dZ` workspace.
- **Fallback** uses a 512-token chunked PyTorch implementation adapted from Verl's fused PPO formulas when the default frontend cannot use the SM90 kernel.

H100 BF16 forward medians from 60 interleaved samples per provider at `H=4096`, `V=131072`. Effective TFLOPS count the common projection work, `2*M*H*V`:

| M | Entropy | CuTe SM90 | cuTile | Verl Torch fallback |
|---:|:--:|---:|---:|---:|
| 2048 | No | **3.12 ms / 706 TFLOPS** | 3.19 ms / 690 TFLOPS | 11.36 ms / 194 TFLOPS |
| 2048 | Yes | **3.13 ms / 703 TFLOPS** | 3.25 ms / 676 TFLOPS | 11.37 ms / 193 TFLOPS |
| 4096 | No | **6.05 ms / 727 TFLOPS** | 6.28 ms / 701 TFLOPS | 22.62 ms / 194 TFLOPS |
| 4096 | Yes | **6.11 ms / 720 TFLOPS** | 6.39 ms / 688 TFLOPS | 22.68 ms / 194 TFLOPS |
| 8192 | No | **12.25 ms / 718 TFLOPS** | 12.41 ms / 709 TFLOPS | 45.35 ms / 194 TFLOPS |
| 8192 | Yes | **12.30 ms / 715 TFLOPS** | 12.70 ms / 693 TFLOPS | 45.59 ms / 193 TFLOPS |
| 16384 | No | **23.84 ms / 738 TFLOPS** | 25.11 ms / 700 TFLOPS | 90.81 ms / 194 TFLOPS |
| 16384 | Yes | **24.09 ms / 730 TFLOPS** | 26.10 ms / 674 TFLOPS | 90.97 ms / 193 TFLOPS |
| 32768 | No | **49.69 ms / 708 TFLOPS** | 54.31 ms / 648 TFLOPS | 180.01 ms / 195 TFLOPS |
| 32768 | Yes | **49.85 ms / 706 TFLOPS** | 55.69 ms / 632 TFLOPS | 179.85 ms / 196 TFLOPS |

Backward medians use 30 interleaved samples per provider. Effective TFLOPS count `6*M*H*V`:

| M | Entropy | CuTe SM90 | cuTile | Verl Torch fallback |
|---:|:--:|---:|---:|---:|
| 8192 | No | **37.93 ms / 696 TFLOPS** | 42.55 ms / 620 TFLOPS | 85.17 ms / 310 TFLOPS |
| 8192 | Yes | **37.89 ms / 696 TFLOPS** | 41.43 ms / 637 TFLOPS | 121.00 ms / 218 TFLOPS |
| 16384 | No | **79.42 ms / 665 TFLOPS** | 83.78 ms / 630 TFLOPS | 166.85 ms / 316 TFLOPS |
| 16384 | Yes | **78.20 ms / 675 TFLOPS** | 84.12 ms / 627 TFLOPS | 239.39 ms / 220 TFLOPS |
| 32768 | No | 163.34 ms / 646 TFLOPS | **163.21 ms / 647 TFLOPS** | 331.08 ms / 319 TFLOPS |
| 32768 | Yes | **160.64 ms / 657 TFLOPS** | 161.73 ms / 653 TFLOPS | 477.51 ms / 221 TFLOPS |

Full forward-and-backward effective TFLOPS count `8*M*H*V`:

| M | Entropy | CuTe SM90 | cuTile | Verl Torch fallback |
|---:|:--:|---:|---:|---:|
| 8192 | No | **50.39 ms / 698 TFLOPS** | 57.15 ms / 616 TFLOPS | 128.76 ms / 273 TFLOPS |
| 8192 | Yes | **50.22 ms / 701 TFLOPS** | 56.86 ms / 619 TFLOPS | 165.63 ms / 212 TFLOPS |
| 16384 | No | **104.77 ms / 672 TFLOPS** | 112.03 ms / 628 TFLOPS | 254.91 ms / 276 TFLOPS |
| 16384 | Yes | **103.65 ms / 679 TFLOPS** | 111.21 ms / 633 TFLOPS | 328.37 ms / 214 TFLOPS |
| 32768 | No | **211.30 ms / 666 TFLOPS** | 221.34 ms / 636 TFLOPS | 508.33 ms / 277 TFLOPS |
| 32768 | Yes | **210.60 ms / 668 TFLOPS** | 221.39 ms / 636 TFLOPS | 655.89 ms / 215 TFLOPS |

B200 measurements for the same shape, using automatic cuTile workspace selection:

| Implementation | Forward | Backward | Full | Peak full memory |
|----|---:|---:|---:|---:|
| cuTile | 2.97 ms | 8.35 ms | 11.36 ms | 2.64 GiB |
| Torch | 5.24 ms | 7.15 ms | 12.85 ms | 7.22 GiB |

There are a couple of ways to apply Liger kernels, depending on the level of customization required.

Using the `AutoLigerKernelForCausalLM` is the simplest approach, as you don't have to import a model-specific patching API. If the model type is supported, the modeling code will be automatically patched using the default settings.

```
from liger_kernel.transformers import AutoLigerKernelForCausalLM

# This AutoModel wrapper class automatically monkey-patches the
# model with the optimized Liger kernels if the model is supported.
model = AutoLigerKernelForCausalLM.from_pretrained("path/to/some/model")
```

Using the [patching APIs](#patching), you can swap Hugging Face models with optimized Liger Kernels.

```
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama

# 1a. Adding this line automatically monkey-patches the model with the optimized Liger kernels
apply_liger_kernel_to_llama()

# 1b. You could alternatively specify exactly which kernels are applied
apply_liger_kernel_to_llama(
  rope=True,
  swiglu=True,
  cross_entropy=True,
  fused_linear_cross_entropy=False,
  rms_norm=False
)

# 2. Instantiate patched model
model = transformers.AutoModelForCausalLM("path/to/llama/model")
```

You can take individual [kernels](https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#model-kernels) to compose your models.

```
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import torch.nn as nn
import torch

model = nn.Linear(128, 256).cuda()

# fuses linear + cross entropy layers together and performs chunk-by-chunk computation to reduce memory
loss_fn = LigerFusedLinearCrossEntropyLoss()

input = torch.randn(4, 128, requires_grad=True, device="cuda")
target = torch.randint(256, (4, ), device="cuda")

loss = loss_fn(model.weight, input, target)
loss.backward()
```

| **AutoModel Variant** | **API** |
|----|----|
| AutoModelForCausalLM | `liger_kernel.transformers.AutoLigerKernelForCausalLM` |

| **Model** | **API** | **Supported Operations** |
|----|----|----|
| Llama4 (Text) & (Multimodal) | `liger_kernel.transformers.apply_liger_kernel_to_llama4` | RMSNorm, LayerNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| LLaMA 2 & 3 | `liger_kernel.transformers.apply_liger_kernel_to_llama` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| LLaMA 3.2-Vision | `liger_kernel.transformers.apply_liger_kernel_to_mllama` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Ministral | `liger_kernel.transformers.apply_liger_kernel_to_ministral` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Mistral | `liger_kernel.transformers.apply_liger_kernel_to_mistral` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Mixtral | `liger_kernel.transformers.apply_liger_kernel_to_mixtral` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Nemotron | `liger_kernel.transformers.apply_liger_kernel_to_nemotron` | ReLUSquared, CrossEntropyLoss, FusedLinearCrossEntropy |
| Pixtral | `liger_kernel.transformers.apply_liger_kernel_to_pixtral` | RoPE, RMSNorm, SwiGLU |
| Gemma1 | `liger_kernel.transformers.apply_liger_kernel_to_gemma` | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Gemma2 | `liger_kernel.transformers.apply_liger_kernel_to_gemma2` | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Gemma3 (Text) | `liger_kernel.transformers.apply_liger_kernel_to_gemma3_text` | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Gemma3 (Multimodal) | `liger_kernel.transformers.apply_liger_kernel_to_gemma3` | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Gemma4 (Text) | `liger_kernel.transformers.apply_liger_kernel_to_gemma4_text` | RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Gemma4 (Multimodal) | `liger_kernel.transformers.apply_liger_kernel_to_gemma4` | RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Paligemma, Paligemma2, & Paligemma2 Mix | `liger_kernel.transformers.apply_liger_kernel_to_paligemma` | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Qwen2, Qwen2.5, & QwQ | `liger_kernel.transformers.apply_liger_kernel_to_qwen2` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Qwen2-VL, & QVQ | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl` | RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Qwen2.5-VL | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_5_vl` | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Qwen3 | `liger_kernel.transformers.apply_liger_kernel_to_qwen3` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Qwen3 MoE | `liger_kernel.transformers.apply_liger_kernel_to_qwen3_moe` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Qwen3.5 | `liger_kernel.transformers.apply_liger_kernel_to_qwen3_5` | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Qwen3.5 MoE (Text) & (Multimodal) | `liger_kernel.transformers.apply_liger_kernel_to_qwen3_5_moe` | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Phi3 & Phi3.5 | `liger_kernel.transformers.apply_liger_kernel_to_phi3` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Granite 3.0 & 3.1 | `liger_kernel.transformers.apply_liger_kernel_to_granite` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss |
| OLMo2 | `liger_kernel.transformers.apply_liger_kernel_to_olmo2` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Olmo3 | `liger_kernel.transformers.apply_liger_kernel_to_olmo3` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| GLM-4 | `liger_kernel.transformers.apply_liger_kernel_to_glm4` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| DeepSeek-V4 | `liger_kernel.transformers.apply_liger_kernel_to_deepseek_v4` | RMSNorm, CrossEntropyLoss, FusedLinearCrossEntropy |
| GPT-OSS | `liger_kernel.transformers.apply_liger_kernel_to_gpt_oss` | RoPE, RMSNorm, CrossEntropyLoss, FusedLinearCrossEntropy |
| InternVL3 | `liger_kernel.transformers.apply_liger_kernel_to_internvl` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| HunyuanV1 | `liger_kernel.transformers.apply_liger_kernel_to_hunyuan_v1_dense` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| HunyuanV1 MoE | `liger_kernel.transformers.apply_liger_kernel_to_hunyuan_v1_moe` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |

- `Fused Linear` kernels combine linear layers with losses, reducing memory usage by up to 80% - ideal for HBM-constrained workloads.
- Other kernels use fusion and in-place techniques for memory and performance optimization.

| **Kernel** | **API** |
|----|----|
| RMSNorm | `liger_kernel.transformers.LigerRMSNorm` |
| Modulated RMSNorm | `liger_kernel.transformers.LigerModulatedRMSNorm` |
| LayerNorm | `liger_kernel.transformers.LigerLayerNorm` |
| RoPE | `liger_kernel.transformers.liger_rotary_pos_emb` |
| SwiGLU | `liger_kernel.transformers.LigerSwiGLUMLP` |
| GeGLU | `liger_kernel.transformers.LigerGEGLUMLP` |
| CrossEntropy | `liger_kernel.transformers.LigerCrossEntropyLoss` |
| Fused Linear CrossEntropy | `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss` |
| Multi Token Attention | `liger_kernel.transformers.LigerMultiTokenAttention` |
| Softmax | `liger_kernel.transformers.LigerSoftmax` |
| Sparsemax | `liger_kernel.transformers.LigerSparsemax` |
| mHC (Hyper-Connections) | `liger_kernel.transformers.LigerMHC` |

| **Kernel** | **API** |
|----|----|
| Fused Linear CPO Loss | `liger_kernel.chunked_loss.LigerFusedLinearCPOLoss` |
| Fused Linear DPO Loss | `liger_kernel.chunked_loss.LigerFusedLinearDPOLoss` |
| Fused Linear ORPO Loss | `liger_kernel.chunked_loss.LigerFusedLinearORPOLoss` |
| Fused Linear SimPO Loss | `liger_kernel.chunked_loss.LigerFusedLinearSimPOLoss` |
| Fused Linear KTO Loss | `liger_kernel.chunked_loss.LigerFusedLinearKTOLoss` |

| **Kernel** | **API** |
|----|----|
| KLDivergence | `liger_kernel.transformers.LigerKLDIVLoss` |
| JSD | `liger_kernel.transformers.LigerJSD` |
| Fused Linear JSD | `liger_kernel.transformers.LigerFusedLinearJSD` |
| TVD | `liger_kernel.transformers.LigerTVDLoss` |

| **Kernel** | **API** |
|----|----|
| Embedding | `liger_kernel.transformers.experimental.LigerEmbedding` |
| Matmul int2xint8 | `liger_kernel.transformers.experimental.matmul` |

- [Contributing Guidelines](https://github.com/linkedin/Liger-Kernel/blob/main/docs/contributing.md)
- [Acknowledgements](https://github.com/linkedin/Liger-Kernel/blob/main/docs/acknowledgement.md)
- [License Information](https://github.com/linkedin/Liger-Kernel/blob/main/docs/license.md)

- [Glows.ai](https://platform.glows.ai/): Sponsoring NVIDIA GPUs for our open source developers.
- [AMD](https://www.amd.com/en.html): Providing AMD GPUs for our AMD CI.
- [Intel](https://www.intel.com/): Providing Intel GPUs for our Intel CI.
- [Modal](https://modal.com/): Free 3000 credits from GPU MODE IRL for our NVIDIA CI.
- [EmbeddedLLM](https://embeddedllm.com/): Making Liger Kernel run fast and stable on AMD.
- [HuggingFace](https://huggingface.co/): Integrating Liger Kernel into Hugging Face Transformers and TRL.
- [Lightning AI](https://lightning.ai/): Integrating Liger Kernel into Lightning Thunder.
- [Axolotl](https://axolotl.ai/): Integrating Liger Kernel into Axolotl.
- [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory): Integrating Liger Kernel into Llama-Factory.

Build

[ ![Build](https://github.com/linkedin/Liger-Kernel/actions/workflows/nvi-ci.yml/badge.svg?branch=main&event=push) ](https://github.com/linkedin/Liger-Kernel/actions/workflows/nvi-ci.yml)

[ ![Build](https://github.com/linkedin/Liger-Kernel/actions/workflows/amd-ci.yml/badge.svg?branch=main&event=push) ](https://github.com/linkedin/Liger-Kernel/actions/workflows/amd-ci.yml)

[ ![Build](https://github.com/linkedin/Liger-Kernel/actions/workflows/intel-ci.yml/badge.svg?branch=main&event=push) ](https://github.com/linkedin/Liger-Kernel/actions/workflows/intel-ci.yml)

[ ![Build](https://github.com/xuedinge233/Liger-Kernel/actions/workflows/ascend_npu_ci.yml/badge.svg?branch=main) ](https://github.com/xuedinge233/Liger-Kernel/actions/workflows/ascend_npu_ci.yml)

- For issues, create a Github ticket in this repository
- For open discussion, join [our discord channel on GPUMode](https://discord.com/channels/1189498204333543425/1275130785933951039)
- For formal collaboration, send an email to Yanning Chen([yannchen@linkedin.com](mailto:yannchen@linkedin.com)) and Zhipeng Wang([zhipwang@linkedin.com](mailto:zhipwang@linkedin.com))

Biblatex entry:

```
@inproceedings{
hsu2025ligerkernel,
title={Liger-Kernel: Efficient Triton Kernels for {LLM} Training},
author={Pin-Lun Hsu and Yun Dai and Vignesh Kothapalli and Qingquan Song and Shao Tang and Siyu Zhu and Steven Shimizu and Shivam Sahni and Haowen Ning and Yanning Chen and Zhipeng Wang},
booktitle={Championing Open-source DEvelopment in ML Workshop @ ICML25},
year={2025},
url={https://openreview.net/forum?id=36SjAIT42G}
}
```

[![Star History Chart](https://camo.githubusercontent.com/b09b02092d51c7a69c6df9e52b13f6c06759d9f9cd38f4a675b2155a1257fcc2/68747470733a2f2f737461722d686973746f72792e646572612e706167652f7376673f7265706f733d6c696e6b6564696e2f4c696765722d4b65726e656c26747970653d44617465)](https://star-history.dera.page/#linkedin/Liger-Kernel&Date)

 [ ↑ Back to Top ↑ ](#readme-top) 
