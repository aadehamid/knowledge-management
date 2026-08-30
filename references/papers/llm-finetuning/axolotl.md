title: GitHub - axolotl-ai-cloud/axolotl: Go ahead and axolotl questions
description: Go ahead and axolotl questions. Contribute to axolotl-ai-cloud/axolotl development by creating an account on GitHub.

# GitHub - axolotl-ai-cloud/axolotl: Go ahead and axolotl questions

 **A Free and Open Source LLM Fine-tuning Framework** \

 [![GitHub License](https://camo.githubusercontent.com/7da0c70d14204a28c582f3fce261e0ef0519de8dbe52f681a7e262c92c2bf1f4/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6963656e73652f61786f6c6f746c2d61692d636c6f75642f61786f6c6f746c2e7376673f636f6c6f723d626c7565)](https://camo.githubusercontent.com/7da0c70d14204a28c582f3fce261e0ef0519de8dbe52f681a7e262c92c2bf1f4/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6963656e73652f61786f6c6f746c2d61692d636c6f75642f61786f6c6f746c2e7376673f636f6c6f723d626c7565) [![codecov](https://camo.githubusercontent.com/ee65975667c51a59343199454ba1634c18a900b0a9faad55675167cb674e24e3/68747470733a2f2f636f6465636f762e696f2f67682f61786f6c6f746c2d61692d636c6f75642f61786f6c6f746c2f6272616e63682f6d61696e2f67726170682f62616467652e737667)](https://codecov.io/gh/axolotl-ai-cloud/axolotl) [![Releases](https://camo.githubusercontent.com/32f4af887842c19e398bbab79d877a9a3d6271f384489dd25455a5e2494f567a/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f72656c656173652f61786f6c6f746c2d61692d636c6f75642f61786f6c6f746c2e737667)](https://github.com/axolotl-ai-cloud/axolotl/releases) \
 [![contributors](https://camo.githubusercontent.com/98f9ecf300bb11f44da979ba9c797dc1df81c78b3e5a283429b119820cf57acd/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f636f6e7472696275746f72732d616e6f6e2f61786f6c6f746c2d61692d636c6f75642f61786f6c6f746c3f636f6c6f723d79656c6c6f77267374796c653d666c61742d737175617265)](https://github.com/axolotl-ai-cloud/axolotl/graphs/contributors) [![GitHub Repo stars](https://camo.githubusercontent.com/eafbec52266784701d6ebb0043b63dc5371f9b46cceba918f311e2c2b6ee9c5e/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f61786f6c6f746c2d61692d636c6f75642f61786f6c6f746c)](https://camo.githubusercontent.com/eafbec52266784701d6ebb0043b63dc5371f9b46cceba918f311e2c2b6ee9c5e/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f61786f6c6f746c2d61692d636c6f75642f61786f6c6f746c) \
 [![discord](https://camo.githubusercontent.com/37727ecbba4188a458fc7012520215f99a7e9509362a15720ca8ca0cb59769ee/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f646973636f72642d3732383964612e7376673f7374796c653d666c61742d737175617265266c6f676f3d646973636f7264)](https://discord.com/invite/HhrNrHJPRb) [![twitter](https://camo.githubusercontent.com/12c6ca5de6f8a018b2caf6e464811239ee07bcce9856650b33b1ac133ba5f90e/68747470733a2f2f696d672e736869656c64732e696f2f747769747465722f666f6c6c6f772f61786f6c6f746c5f61693f7374796c653d736f6369616c)](https://twitter.com/axolotl_ai) [![google-colab](https://camo.githubusercontent.com/eff96fda6b2e0fff8cdf2978f89d61aa434bb98c00453ae23dd0aab8d1451633/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb) \

- 2026/08:
    - New model support has been added in Axolotl for [Muse Glimmer](https://docs.axolotl.ai/docs/models/muse-glimmer.html), [North Micro Vision Instruct](https://docs.axolotl.ai/docs/models/cohere-north-micro-vision-instruct.html) and [Shieldstral](https://docs.axolotl.ai/docs/models/shieldstral.html).
- 2026/07:
    - [NVFP4 (4-bit) MoE LoRA training](https://docs.axolotl.ai/docs/nvfp4_lora.html) is now supported via ScatterMoE (W4A16) and SonicMoE (W4A4), including adapter merge back into a plain NVFP4 checkpoint.
- 2026/06:
    - [Expert Parallelism (EP)](https://docs.axolotl.ai/docs/nd_parallelism.html) for distributed MoE training via DeepEP, remote training through [Tinker-compatible APIs](https://github.com/axolotl-ai-cloud/axolotl/pull/3614), [Context Parallelism for hybrid SSM models](https://github.com/axolotl-ai-cloud/axolotl/pull/3572) (Nemotron-H, Falcon-H1, Bamba), [BitNet 1.58-bit](https://github.com/axolotl-ai-cloud/axolotl/pull/3634) fine-tuning, and a [multimodal assistant-only loss-masking fix](https://github.com/axolotl-ai-cloud/axolotl/pull/3625).
- 2026/04:
    - New model support has been added in Axolotl for [Mistral Medium 3.5](https://docs.axolotl.ai/docs/models/mistral-medium-3_5.html) and [Gemma 4](https://docs.axolotl.ai/docs/models/gemma4.html).
    - New RL and kernels: [Async GRPO](https://github.com/axolotl-ai-cloud/axolotl/pull/3486) (up to 58% faster steps), [Flash Attention 4](https://docs.axolotl.ai/docs/attention.html#flash-attention), [NeMo Gym](https://github.com/axolotl-ai-cloud/axolotl/pull/3516), and [EBFT](https://github.com/axolotl-ai-cloud/axolotl/pull/3527).
    - Axolotl is now [uv-first](https://github.com/axolotl-ai-cloud/axolotl/pull/3545) and has [SonicMoE fused LoRA](https://github.com/axolotl-ai-cloud/axolotl/pull/3519) support.
- 2026/03:
    - New model support has been added in Axolotl for [Mistral Small 4](https://docs.axolotl.ai/docs/models/mistral4.html), [Qwen3.5, Qwen3.5 MoE](https://docs.axolotl.ai/docs/models/qwen3.5.html), [GLM-4.7-Flash](https://docs.axolotl.ai/docs/models/glm47-flash.html), [GLM-4.6V](https://docs.axolotl.ai/docs/models/glm46v.html), and [GLM-4.5-Air](https://docs.axolotl.ai/docs/models/glm45.html).
    - [MoE expert quantization](https://docs.axolotl.ai/docs/expert_quantization.html) support (via `quantize_moe_experts: true`) greatly reduces VRAM when training MoE models (FSDP2 compat).

> [!NOTE]- Expand older updates
> - 2026/02:
>     - [ScatterMoE LoRA](https://github.com/axolotl-ai-cloud/axolotl/pull/3410) support. LoRA fine-tuning directly on MoE expert weights using custom Triton kernels.
>     - Axolotl now has support for [SageAttention](https://github.com/axolotl-ai-cloud/axolotl/pull/2823) and [GDPO](https://github.com/axolotl-ai-cloud/axolotl/pull/3353) (Generalized DPO).
> - 2026/01:
>     - New integration for [EAFT](https://github.com/axolotl-ai-cloud/axolotl/pull/3366) (Entropy-Aware Focal Training), weights loss by entropy of the top-k logit distribution, and [Scalable Softmax](https://github.com/axolotl-ai-cloud/axolotl/pull/3338), improves long context in attention.
> - 2025/12:
>     - Axolotl now includes support for [Kimi-Linear](https://docs.axolotl.ai/docs/models/kimi-linear.html), [Plano-Orchestrator](https://docs.axolotl.ai/docs/models/plano.html), [MiMo](https://docs.axolotl.ai/docs/models/mimo.html), [InternVL 3.5](https://docs.axolotl.ai/docs/models/internvl3_5.html), [Olmo3](https://docs.axolotl.ai/docs/models/olmo3.html), [Trinity](https://docs.axolotl.ai/docs/models/trinity.html), and [Ministral3](https://docs.axolotl.ai/docs/models/ministral3.html).
>     - [Distributed Muon Optimizer](https://github.com/axolotl-ai-cloud/axolotl/pull/3264) support has been added for FSDP2 pretraining.
> - 2025/10: New model support has been added in Axolotl for: [Qwen3 Next](https://docs.axolotl.ai/docs/models/qwen3-next.html), [Qwen2.5-vl, Qwen3-vl](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/qwen2_5-vl), [Qwen3, Qwen3MoE](https://docs.axolotl.ai/docs/models/qwen3.html), [Granite 4](https://docs.axolotl.ai/docs/models/granite4.html), [HunYuan](https://docs.axolotl.ai/docs/models/hunyuan.html), [Magistral 2509](https://docs.axolotl.ai/docs/models/magistral/vision.html), [Apertus](https://docs.axolotl.ai/docs/models/apertus.html), and [Seed-OSS](https://docs.axolotl.ai/docs/models/seed-oss.html).
> - 2025/09: Axolotl now has text diffusion training. Read more [here](https://github.com/axolotl-ai-cloud/axolotl/tree/main/src/axolotl/integrations/diffusion).
> - 2025/08: QAT has been updated to include NVFP4 support. See [PR](https://github.com/axolotl-ai-cloud/axolotl/pull/3107).
> - 2025/07:
>     - ND Parallelism support has been added into Axolotl. Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes. Check out the [blog post](https://huggingface.co/blog/accelerate-nd-parallel) for more info.
>     - Axolotl adds more models: [GPT-OSS](https://docs.axolotl.ai/docs/models/gpt-oss.html), [Gemma 3n](https://docs.axolotl.ai/docs/models/gemma3n.html), [Liquid Foundation Model 2 (LFM2)](https://docs.axolotl.ai/docs/models/LiquidAI.html), and [Arcee Foundation Models (AFM)](https://docs.axolotl.ai/docs/models/arcee.html).
>     - FP8 finetuning with fp8 gather op is now possible in Axolotl via `torchao`. Get started [here](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)!
>     - [Voxtral](https://docs.axolotl.ai/docs/models/voxtral.html), [Magistral 1.1](https://docs.axolotl.ai/docs/models/magistral.html), and [Devstral](https://docs.axolotl.ai/docs/models/devstral.html) with mistral-common tokenizer support has been integrated in Axolotl!
>     - TiledMLP support for single-GPU to multi-GPU training with DDP, DeepSpeed and FSDP support has been added to support Arctic Long Sequence Training. (ALST). See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst) for using ALST with Axolotl!
> - 2025/06: Magistral with mistral-common tokenizer support has been added to Axolotl. See [docs](https://docs.axolotl.ai/docs/models/magistral.html) to start training your own Magistral models with Axolotl!
> - 2025/05: Quantization Aware Training (QAT) support has been added to Axolotl. Explore the [docs](https://docs.axolotl.ai/docs/qat.html) to learn more!
> - 2025/04: Llama 4 support has been added in Axolotl. See [docs](https://docs.axolotl.ai/docs/models/llama-4.html) to start training your own Llama 4 models with Axolotl's linearized version!
> - 2025/03: Axolotl has implemented Sequence Parallelism (SP) support. Read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html) to learn how to scale your context length when fine-tuning.
> - 2025/03: (Beta) Fine-tuning Multimodal models is now supported in Axolotl. Check out the [docs](https://docs.axolotl.ai/docs/multimodal.html) to fine-tune your own!
> - 2025/02: Axolotl has added LoRA optimizations to reduce memory usage and improve training speed for LoRA and QLoRA in single GPU and multi-GPU training (DDP and DeepSpeed). Jump into the [docs](https://docs.axolotl.ai/docs/lora_optims.html) to give it a try.
> - 2025/02: Axolotl has added GRPO support. Dive into our [blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code) and have some fun!
> - 2025/01: Axolotl has added Reward Modelling / Process Reward Modelling fine-tuning support. See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).

Axolotl is a free and open-source tool designed to streamline post-training and fine-tuning for the latest large language models (LLMs).

Features:

- **Multiple Model Support**: Train various models like GPT-OSS, LLaMA, Mistral, Mixtral, Pythia, and many more models available on the Hugging Face Hub.
- **Multimodal Training**: Fine-tune vision-language models (VLMs) including LLaMA-Vision, Qwen2-VL, Pixtral, LLaVA, SmolVLM2, GLM-4.6V, InternVL 3.5, Gemma 3n, PaddleOCR-VL, Muse Glimmer, and audio models like Voxtral with image, video, and audio support.
- **Training Methods**: Full fine-tuning, LoRA, QLoRA, GPTQ, QAT (int8/int4/FP8/NVFP4/MXFP4), FP8 mixed-precision training, NVFP4/MXFP4 MoE LoRA, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO, GDPO), and Reward Modelling (RM) / Process Reward Modelling (PRM).
- **Easy Configuration**: Re-use a single YAML configuration file across the full fine-tuning pipeline: dataset preprocessing, training, evaluation, quantization, and inference.
- **Performance Optimizations**: [Multipacking](https://docs.axolotl.ai/docs/multipack.html), [Flash Attention 2/3/4](https://docs.axolotl.ai/docs/attention.html#flash-attention), [Xformers](https://docs.axolotl.ai/docs/attention.html#xformers), [Flex Attention](https://docs.axolotl.ai/docs/attention.html#flex-attention), [SageAttention](https://docs.axolotl.ai/docs/attention.html#sageattention), [Liger Kernel](https://docs.axolotl.ai/docs/custom_integrations.html#liger-kernels), [Cut Cross Entropy](https://docs.axolotl.ai/docs/custom_integrations.html#cut-cross-entropy), [ScatterMoE](https://docs.axolotl.ai/docs/custom_integrations.html#kernels-integration), [Sequence Parallelism (SP)](https://docs.axolotl.ai/docs/sequence_parallelism.html), [LoRA optimizations](https://docs.axolotl.ai/docs/lora_optims.html), [Multi-GPU training (FSDP1, FSDP2, DeepSpeed)](https://docs.axolotl.ai/docs/multi-gpu.html), [Multi-node training (Torchrun, Ray)](https://docs.axolotl.ai/docs/multi-node.html), and many more!
- **Flexible Dataset Handling**: Load from local, HuggingFace, and cloud (S3, Azure, GCP, OCI) datasets.
- **Cloud Ready**: We ship [Docker images](https://hub.docker.com/u/axolotlai) and also [PyPI packages](https://pypi.org/project/axolotl/) for use on cloud platforms and local hardware.

**Requirements**:

- NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
- Python >\=3.11 (3.12 recommended)
- PyTorch ≥2.11.0 (2.12.1 recommended)

[![Open In Colab](https://camo.githubusercontent.com/eff96fda6b2e0fff8cdf2978f89d61aa434bb98c00453ae23dd0aab8d1451633/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb#scrollTo=msOCO4NRmRLa)

```
# install uv if you don't already have it installed (restart shell after)
curl -LsSf https://astral.sh/uv/install.sh | sh

# change depending on system
export UV_TORCH_BACKEND=cu130

# create a new virtual environment
uv venv --python 3.12
source .venv/bin/activate

uv pip install torch==2.12.1 torchvision
uv pip install --no-build-isolation axolotl[deepspeed]

# Download example axolotl configs, deepspeed configs
axolotl fetch examples
axolotl fetch deepspeed_configs  # OPTIONAL
```

Installing with Docker can be less error prone than installing in your own environment.

```
docker run --gpus '"all"' --ipc=host --rm -it axolotlai/axolotl:main-latest
```

Other installation approaches are described [here](https://docs.axolotl.ai/docs/installation.html).

> [!NOTE]- Details
> - [RunPod](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)
> - [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer_community&utm_campaign=template_launch_axolotl&utm_content=readme)
> - [PRIME Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=axolotl&location=Cheapest&security=Cheapest&show_spot=true)
> - [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl)
> - [Novita](https://novita.ai/gpus-console?templateId=311)
> - [JarvisLabs.ai](https://jarvislabs.ai/templates/axolotl)
> - [Latitude.sh](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)

```
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

That's it! Check out our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a more detailed walkthrough.

- [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Detailed setup instructions for different environments
- [Support Matrix](https://docs.axolotl.ai/docs/support-matrix.html) - Feature support, compatibility, and known gaps
- [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Full configuration options and examples
- [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets from various sources
- [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats and how to use them
- [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
- [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
- [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated code documentation
- [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently asked questions

Axolotl ships with built-in documentation optimized for AI coding agents (Claude Code, Cursor, Copilot, etc.). These docs are bundled with the pip package, no repo clone needed.

```
# Show overview and available training methods
axolotl agent-docs

# Topic-specific references
axolotl agent-docs sft                 # supervised fine-tuning
axolotl agent-docs grpo                # GRPO online RL
axolotl agent-docs preference_tuning   # DPO, KTO, ORPO, SimPO
axolotl agent-docs reward_modelling    # outcome and process reward models
axolotl agent-docs pretraining         # continual pretraining
axolotl agent-docs --list              # list all topics

# Dump config schema for programmatic use
axolotl config-schema
axolotl config-schema --field adapter
```

If you're working with the source repo, agent docs are also available at `docs/agents/` and the project overview is in `AGENTS.md`.

- Join our [Discord community](https://discord.gg/HhrNrHJPRb) for support
- Check out our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) directory
- Read our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
- Need dedicated support? Please contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai) for options

Contributions are welcome! Please see our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

Axolotl has opt-out telemetry that helps us understand how the project is being used and prioritize improvements. We collect basic system information, model types, and error rates, never personal data or file paths. Telemetry is enabled by default. To disable it, set AXOLOTL\_DO\_NOT\_TRACK\=1. For more details, see our [telemetry documentation](https://docs.axolotl.ai/docs/telemetry.html).

Interested in sponsoring? Contact us at [wing@axolotl.ai](mailto:wing@axolotl.ai)

If you use Axolotl in your research or projects, please cite it as follows:

```
@software{axolotl,
  title = {Axolotl: Open Source LLM Post-Training},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/axolotl-ai-cloud/axolotl/blob/main/LICENSE) file for details.
