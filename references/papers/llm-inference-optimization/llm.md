title: LLM Inference Handbook
description: A practical handbook for engineers building, optimizing, scaling and operating LLM inference systems in production.
keywords: LLM inference guide, LLM inference handbook, LLM inference book, LLM inference best practices, Inference, LLM, LLM inference, AI inference, GenAI inference, Inference optimization, inference techniques, LLM fast inference, Inference platform, inference operations, Efficient generative LLM inference, distributed LLM inference

> [llms.txt](https://bentoml.com/llms.txt)

# LLM Inference Handbook

*LLM Inference Handbook* is your technical glossary, guidebook, and reference - all in one. It covers everything you need to know about LLM inference, from core concepts and performance metrics (e.g., [Time to First Token and Tokens per Second](https://bentoml.com/llm-inference-basics/llm-inference-metrics/)), to optimization techniques (e.g., [continuous batching](https://bentoml.com/inference-optimization/static-dynamic-continuous-batching/) and [prefix caching](https://bentoml.com/inference-optimization/prefix-caching/)), [GPU architecture](https://bentoml.com/kernel-optimization/gpu-architecture-fundamentals/), and deployment patterns like [BYOC](https://bentoml.com/getting-started/bring-your-own-cloud/) and [on-prem](https://bentoml.com/getting-started/on-prem-llms/).

- Practical guidance for deploying, scaling, and operating LLMs in production.
- Explore concepts with interactive calculators, simulators, and visual tools.
- Boost performance with optimization techniques tailored to your use case.
- Continuously updated with the latest best practices and field-tested insights.

## Motivation {#motivation}

We wrote this handbook to solve a common problem facing developers: LLM inference knowledge is often fragmented; it’s buried in academic papers, scattered across vendor blogs, hidden in GitHub issues, or tossed around in Discord threads. Worse, much of it assumes you already understand half the stack.

There aren’t many resources that bring it all together — like how [inference differs from training](https://bentoml.com/llm-inference-basics/training-inference-differences/), why [goodput matters more than raw throughput](https://bentoml.com/llm-inference-basics/llm-inference-metrics/#goodput) for meeting SLOs, or how [prefill-decode disaggregation](https://bentoml.com/inference-optimization/prefill-decode-disaggregation/) works in practice.

So we started pulling it all together.

## Who this is for {#who-this-is-for}

This handbook is for engineers deploying, scaling or operating LLMs in production, whether you're fine-tuning a small open model or running large-scale deployments on your own stack.

If your goal is to make LLM inference faster, cheaper, or more reliable, this handbook is for you.

## How to use this {#how-to-use-this}

You can read it start-to-finish or treat it like a lookup table. There’s no wrong way to navigate. We’ll keep updating the handbook as the field evolves, because LLM inference is changing fast, and what works today may not be best tomorrow.

## Interactive tools {#interactive-tools}

This handbook provides various interactive tools to help you learn by trying the concepts directly:

- [LLM Inference Visualizer](https://bentoml.com/llm-inference-basics/what-is-llm-inference/): Walk through the request lifecycle and see how tokens flow through prefill and decode.
- [LLM Lifecycle Visualizer](https://bentoml.com/llm-inference-basics/training-inference-differences/): See where training and inference sit in the model lifecycle, and how inference runs on every request.
- [Token-by-Token Decode Loop](https://bentoml.com/llm-inference-basics/how-does-llm-inference-work/#decode): Step through autoregressive decoding and watch each new token extend the sequence and KV cache.
- [Latency Timeline Visualizer](https://bentoml.com/llm-inference-basics/how-does-llm-inference-work/#decode): See how every decode step is followed by detokenization, and which stages TTFT, ITL, and E2EL span.
- [Context Window Simulator](https://bentoml.com/llm-inference-basics/how-does-llm-inference-work/#what-is-a-context-window-and-how-does-it-work-in-llm-inference): See how the full conversation is re-sent each turn and fills the context window.
- [Latency Metrics Playground](https://bentoml.com/llm-inference-basics/llm-inference-metrics/#latency): Explore TTFT, E2EL, TPOT, and SLO-based goodput.
- [Top-p vs Top-k Filter](https://bentoml.com/model-interaction/inference-parameters/#top-p-and-top-k-sampling): Compare how each filter handles peaky, mixed, and flat distributions.
- [Model Explorer](https://bentoml.com/getting-started/choosing-the-right-model/): Browse popular open-source LLMs and compare their architecture, scale, context, and typical GPU deployment.
- [GPU Comparison Table](https://bentoml.com/getting-started/choosing-the-right-gpu/#matching-gpus-to-open-source-llms): Match popular open-source LLMs to suitable NVIDIA and AMD GPUs.
- [GPU Memory Calculator](https://bentoml.com/getting-started/calculating-gpu-memory-for-llms/): Estimate VRAM requirements for serving an LLM.
- [Quantization Memory Impact Visualizer](https://bentoml.com/model-preparation/llm-quantization/#quantization-formats): Compare weight memory across quantization formats.
- [Batching Strategy Simulator](https://bentoml.com/inference-optimization/static-dynamic-continuous-batching/): Compare static, dynamic, and continuous batching behavior.
- [Chunked Prefill Scheduler](https://bentoml.com/inference-optimization/static-dynamic-continuous-batching/#chunked-prefill): See how a whole prefill stalls active decodes, and how chunking lets them continue.
- [KV Cache Memory Calculator](https://bentoml.com/inference-optimization/kv-cache-offloading/#how-to-calculate-the-kv-cache-size): Estimate how much memory the KV cache consumes.
- [GPU Execution and Memory Map](https://bentoml.com/kernel-optimization/gpu-architecture-fundamentals/): Visualize how threads, warps, SMs, and the GPU memory hierarchy fit together.
- [GPU Memory Explorer](https://bentoml.com/kernel-optimization/gpu-architecture-fundamentals/gpu-memory/): Understand different memory tiers and see how reuse scope changes as data moves.
- [Streaming Multiprocessor Explorer](https://bentoml.com/kernel-optimization/gpu-architecture-fundamentals/streaming-multiprocessors/#what-an-sm-contains): Explore the units inside an SM and see how many of each an H100 has.
- [Warp Scheduler Visualizer](https://bentoml.com/kernel-optimization/gpu-architecture-fundamentals/streaming-multiprocessors/#how-warp-scheduling-hides-latency): Compare too few, enough, and extra resident warps on a scheduler timeline.

## Contributing {#contributing}

We welcome contributions! If you spot an error, have suggestions for improvements, or want to add new topics, please open an issue or submit a pull request on our [GitHub repository](https://github.com/modular/llm-inference-handbook).
