title: Train an LLM on NVIDIA Blackwell with Unsloth—and Scale for Production | NVIDIA Technical Blog
description: Fine-tuning and reinforcement learning \(RL\) for large language models \(LLMs\) require advanced expertise and complex workflows, making them out of reach for many.

[Agentic AI / Generative AI](https://developer.nvidia.com/blog/category/generative-ai/)

# Train an LLM on NVIDIA Blackwell with Unsloth—and Scale for Production

![](https://developer-blogs.nvidia.com/wp-content/uploads/2025/10/cloud-computing-1024x576-png.webp){width=1024 height=576}

> [!NOTE]- Details
> [Unsloth](https://github.com/unslothai/unsloth) is an open source framework that simplifies and accelerates LLM fine-tuning and reinforcement learning with custom Triton kernels, delivering 2x faster training throughput and 70% less VRAM usage without accuracy loss.On NVIDIA Blackwell GPUs, Unsloth achieves a 2x increase in training speed, 70% VRAM reduction for models up to 70B\+ parameters, and 12x longer context windows, enabling fine-tuning of models with up to 40 billion parameters on a single GPU.The framework supports popular models such as Llama, gpt-oss, and DeepSeek, and is optimized for NVFP4 precision on Blackwell hardware.Unsloth workflows run locally on NVIDIA GeForce RTX 50 Series, RTX PRO 6000 Blackwell Series, and NVIDIA DGX Spark systems, then scale seamlessly to NVIDIA DGX Cloud and NVIDIA Cloud Partners for production workloads without code changes.Next StepsRead the [step-by-step guide to fine-tuning LLMs with NVIDIA Blackwell GPUs and Unsloth](https://docs.unsloth.ai/basics/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth) to begin local experimentation.Explore how to [install the software on NVIDIA DGX Spark](https://build.nvidia.com/spark/unsloth) for developer workstation deployments.Powered by NVIDIA Nemotron. AI-generated content may summarize information incompletely. Verify important information. 

Fine-tuning and reinforcement learning (RL) for [large language models (LLMs)](https://www.nvidia.com/en-us/glossary/large-language-models/) require advanced expertise and complex workflows, making them out of reach for many. The open source [Unsloth project](https://github.com/unslothai/unsloth) changes that by streamlining the process, making it easier for individuals and small teams to explore LLM customization. When paired with the efficiency and throughput of the NVIDIA Blackwell GPUs, this combination helps democratize access to LLM development, opening the door for a wider community of practitioners to innovate.\
 \
This post explains how developers can train custom LLMs locally on [NVIDIA RTX PRO 6000 Blackwell Series](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/), [GeForce RTX 50 Series](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/), and [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) using Unsloth. It also covers how these same workflows scale seamlessly into Blackwell-powered cloud instances, such as [NVIDIA DGX Cloud](https://developer.nvidia.com/dgx-cloud) and those from [NVIDIA Cloud Partners](https://www.nvidia.com/en-us/data-center/gpu-cloud-computing/partners/), for production workloads.

## What is Unsloth?[](#what_is_unsloth) {#what_is_unsloth}

[Unsloth](https://unsloth.ai/about) is an open source framework that simplifies and accelerates LLM fine-tuning and RL. It uses custom Triton kernels and algorithms to deliver:

- 2x faster training throughput
- 70% less VRAM usage
- No accuracy loss

It supports popular models such as Llama, gpt-oss, and DeepSeek, and is now optimized for NVIDIA Blackwell GPUs with [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) precision.\
 \
With support from the NVIDIA DGX Cloud AI team, Unsloth extends from consumer GPUs, such as the GeForce RTX 50 Series, RTX PRO 6000 Blackwell Series, and NVIDIA GB10-based developer workstations (such as the NVIDIA DGX Spark), to enterprise-class [NVIDIA HGX B200](https://www.nvidia.com/en-us/data-center/hgx/) and [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) systems. This makes fine-tuning accessible to everyone.

## How does Unsloth perform on NVIDIA Blackwell? [](#how_does_unsloth_perform_on_nvidia_blackwell) {#how_does_unsloth_perform_on_nvidia_blackwell }

[Unsloth benchmarks](https://docs.unsloth.ai/basics/unsloth-benchmarks) show that, with NVIDIA Blackwell, it delivers significant gains compared to other optimized setups, including Flash Attention 2. Specifically, it delivers:

- 2x increase in training speed
- 70% VRAM reduction (even for 70B\+ parameter models)
- 12x longer context windows

These results mean that you can now fine-tune models with as many as 40 billion parameters on a single Blackwell GPU.

Test setup: NVIDIA GeForce RTX 5090 GPU with 32 GB of VRAM, Alpaca dataset, batch size \= 2, gradient accumulation \= 4, rank \= 32, QLoRA applied on all linear layers.

Model

VRAM

Unsloth speed

VRAM reduction

Longer context

Hugging Face \+ FA2

*Table 1. Performance benchmarks for Unsloth on a GeForce RTX 5090 GPU*

VRAM

Unsloth context length

Hugging Face \+ FA2 context length

*Table 2. Detailed benchmarks for different context lengths for Unsloth on a GeForce RTX 5090 GPU*

## How to set up Unsloth on NVIDIA GPUs[](#how_to_set_up_unsloth_on_nvidia_gpus) {#how_to_set_up_unsloth_on_nvidia_gpus}

Unsloth setup is easy, whether you prefer a quick pip install, an isolated virtual environment, or a containerized Docker deployment. Try the following examples on any Blackwell generation GPU, including the GeForce RTX 50 Series.

```

pip install unsloth
```

### Running a 20B model[](#running_a_20b_model) {#running_a_20b_model}

The following example shows what it might look like to run the gpt-oss-20b model:

```

from unsloth import FastLanguageModel
import torch
max_seq_length = 1024
# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit", # 20B model using bitsandbytes 4bit quantization
    "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    "unsloth/gpt-oss-20b", # 20B model using MXFP4 format
    "unsloth/gpt-oss-120b",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)
```

### Docker deployment[](#docker_deployment) {#docker_deployment}

Unsloth also offers a [prebuilt Docker image](https://hub.docker.com/r/unsloth/unsloth), which is supported in NVIDIA Blackwell GPUs. 

Note that the Docker container requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to be installed on your host system.\
Before running the following command, fill in your specific information:

```

docker run -d -e JUPYTER_PASSWORD="mypassword" \
  -p 8888:8888 -p 2222:22 \
  -v $(pwd)/work:/workspace/work \
  --gpus all \
  unsloth/unsloth
```

### Using an isolated environment[](#using_an_isolated_environment) {#using_an_isolated_environment}

Issue the following commands from the shell to install Unsloth using Python:

```

python -m venv unsloth
source unsloth/bin/activate
pip install unsloth
```

Note: Depending on your system, you may need to use `pip3` / `pip3.13` and `python3` / `python3.13`.

### Handling issues with xFormers [](#handling_issues_with_xformers) {#handling_issues_with_xformers }

If you encounter issues with xFormers, build from source. 

First, uninstall any existing xFormers:

```

pip uninstall xformers -y
```

Next, clone and build:

```

pip install ninja
export TORCH_CUDA_ARCH_LIST="12.0"
git clone --depth=1 https://github.com/facebookresearch/xformers --recursive
cd xformers && python setup.py install && cd ..
```

### Using uv[](#using_uv) {#using_uv}

If you prefer to use `uv`, install Unsloth using the following command:

```

uv pip install unsloth
```

While Unsloth enables local experimentation with 20B and 40B models on a single Blackwell GPU, the same workflows are fully portable to NVIDIA DGX Cloud and NVIDIA Cloud Partners. This enables scaling to clusters of Blackwell GPUs for fine-tuning 70B\+ models, reinforcement learning, and enterprise workloads without changing a line of code.

## Get started transforming LLM training runs[](#get_started_transforming_llm_training_runs) {#get_started_transforming_llm_training_runs}

From experimentation to production, [NVIDIA DGX Cloud](https://developer.nvidia.com/dgx-cloud) and [NVIDIA Cloud Partners](https://www.nvidia.com/en-us/data-center/gpu-cloud-computing/partners/) deliver the power to train and fine-tune at any scale—combining elastic compute, enterprise storage, and real-time monitoring in fully managed AI environments optimized for NVIDIA GPUs.

According to Unsloth Co-Founders Daniel and Michael Han, “AI shouldn’t be an exclusive club. The next great AI breakthrough could come from anywhere—students, individual researchers, or small startups. Unsloth is here to ensure they have the tools they need.”

Start locally on your NVIDIA GeForce RTX 50 Series GPU, NVIDIA RTX PRO 6000 Blackwell Series GPU, or NVIDIA DGX Spark system to fine-tune models with Unsloth. Then scale seamlessly with NVIDIA DGX Cloud or an NVIDIA Cloud Partner to harness clusters of Blackwell GPUs with enterprise-grade reliability and visibility—all without compromise. Check out the [step-by-step guide to fine-tuning LLMs with NVIDIA Blackwell GPUs and Unsloth](https://docs.unsloth.ai/basics/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth), and how to [install the software on NVIDIA DGX Spark.](https://build.nvidia.com/spark/unsloth)

[  Discuss (1) ](#entry-content-comments)

 **About Paul Abruzzo**  \
Paul Abruzzo is a product and engineering lead for NVIDIA DGX Cloud Engineering. His work is focused on Forward Deployed Engineering and Solutions Architecture, working directly with open-source generative AI frameworks and the world's largest AI Native organizations to develop feature support and performance optimization for novel NVIDIA technologies. Before joining NVIDIA, Paul helped build foundational cloud technologies at Oracle, AWS, and Apple. He carries a love for disruptive technology, clean data halls, industrial design, and automobiles that don't shift their own gears. 

 **About Jason Perlow**  \
Jason Perlow is a senior member of the NVIDIA DGX Cloud team, focusing on technical documentation to support AI and cloud developers. Jason was formerly the Editorial Director of The Linux Foundation and has held positions at Microsoft and IBM as a trusted advisor on cloud and datacenter technology. 

 **About Brian Carpenter**  \
Brian Carpenter leads open source software, technical community, and partnership initiatives for NVIDIA DGX Cloud. He brings deep experience in product, strategy, and developer relations from senior roles across the enterprise technology landscape. 
