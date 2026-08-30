title: Learn Inference: inference engineering, explained interactively
description: An interactive guide to inference engineering: how generative AI models are served in production, from attention kernels to multi-cloud capacity.
keywords: inference engineering, LLM inference, GPU inference, KV cache, speculative decoding, quantization, vLLM, SGLang, roofline model, model serving

# Learn Inference: inference engineering, explained interactively

## Learn\
Inference

Serving generative\
models in production

[Start reading](https://learn-inference.com/chapters/inference) [Glossary](https://learn-inference.com/chapters/glossary) [Further reading](https://learn-inference.com/chapters/reading) [The source book](https://www.baseten.co/inference-engineering/)

## Everything after training

Training teaches a model what it knows. Inference is everything that happens afterward, every time somebody uses it, and it is where the bill actually lands. Serving a generative model well means working across a strange range of the stack: attention kernels at one end, GPU procurement across three clouds at the other.

This is an interactive companion to *Inference Engineering* by Philip Kiely. It follows the book’s structure and covers the same ground. I rewrote the explanations and built simulators for the parts you grasp faster by turning a dial than by reading a paragraph.

## Try one

This one is from chapter 1. Press run, then drag the sliders to feel the difference between starting fast and finishing fast.

*Figure 1.4. Time to first token versus tokens per second. Two numbers, two phases, two bottlenecks. A response that starts instantly and trickles can feel faster than one that pauses and then dumps, even when the second finishes first.Illustrative numbers*

### [0. Inference](https://learn-inference.com/chapters/inference)

- [0.1 Two phases, two disciplines](https://learn-inference.com/chapters/inference/two-phases)
- [0.2 The three layers](https://learn-inference.com/chapters/inference/three-layers)
- [0.3 Six techniques that define the runtime](https://learn-inference.com/chapters/inference/runtime-techniques)
- [0.4 Scale changes the problem](https://learn-inference.com/chapters/inference/scale-changes-problem)
- [0.5 Where to put the abstraction](https://learn-inference.com/chapters/inference/abstraction)
- [0.6 A map of what follows](https://learn-inference.com/chapters/inference/map)

### [1. Prerequisites](https://learn-inference.com/chapters/prerequisites)

- [1.1 Scale and specialization](https://learn-inference.com/chapters/prerequisites/scale-and-specialization)
- [1.2 About your app](https://learn-inference.com/chapters/prerequisites/about-your-app)
- [1.3 Model selection](https://learn-inference.com/chapters/prerequisites/model-selection)
- [1.4 Measuring latency and throughput](https://learn-inference.com/chapters/prerequisites/latency-throughput)

### [2. Models](https://learn-inference.com/chapters/models)

- [2.1 Neural networks](https://learn-inference.com/chapters/models/neural-networks)
- [2.2 LLM inference mechanics](https://learn-inference.com/chapters/models/llm-mechanics)
- [2.3 Image generation inference mechanics](https://learn-inference.com/chapters/models/image-mechanics)
- [2.4 Calculating inference bottlenecks](https://learn-inference.com/chapters/models/bottlenecks)
- [2.5 Optimizing attention](https://learn-inference.com/chapters/models/optimizing-attention)

### [3. Hardware](https://learn-inference.com/chapters/hardware)

- [3.1 GPU architecture](https://learn-inference.com/chapters/hardware/gpu-architecture)
- [3.2 GPU architecture generations](https://learn-inference.com/chapters/hardware/generations)
- [3.3 Instances](https://learn-inference.com/chapters/hardware/instances)
- [3.4 Other datacenter accelerator options](https://learn-inference.com/chapters/hardware/other-accelerators)
- [3.5 Local inference](https://learn-inference.com/chapters/hardware/local-inference)

### [4. Software](https://learn-inference.com/chapters/software)

- [4.1 CUDA](https://learn-inference.com/chapters/software/cuda)
- [4.2 Deep learning frameworks and libraries](https://learn-inference.com/chapters/software/frameworks)
- [4.3 Inference engines](https://learn-inference.com/chapters/software/engines)
- [4.4 NVIDIA Dynamo](https://learn-inference.com/chapters/software/dynamo)
- [4.5 Performance benchmarking and load testing](https://learn-inference.com/chapters/software/benchmarking)

### [5. Techniques](https://learn-inference.com/chapters/techniques)

- [5.1 Quantization](https://learn-inference.com/chapters/techniques/quantization)
- [5.2 Speculative decoding](https://learn-inference.com/chapters/techniques/speculative-decoding)
- [5.3 Caching](https://learn-inference.com/chapters/techniques/caching)
- [5.4 Model parallelism](https://learn-inference.com/chapters/techniques/parallelism)
- [5.5 Disaggregation](https://learn-inference.com/chapters/techniques/disaggregation)

### [6. Modalities](https://learn-inference.com/chapters/modalities)

- [6.1 Vision language models](https://learn-inference.com/chapters/modalities/vlms)
- [6.2 Embedding models](https://learn-inference.com/chapters/modalities/embeddings)
- [6.3 ASR models](https://learn-inference.com/chapters/modalities/asr)
- [6.4 TTS models](https://learn-inference.com/chapters/modalities/tts)
- [6.5 Image generation models](https://learn-inference.com/chapters/modalities/image-models)
- [6.6 Video generation models](https://learn-inference.com/chapters/modalities/video-models)

### [7. Production](https://learn-inference.com/chapters/production)

- [7.1 Containerization](https://learn-inference.com/chapters/production/containerization)
- [7.2 Autoscaling](https://learn-inference.com/chapters/production/autoscaling)
- [7.3 Multi-cloud capacity management](https://learn-inference.com/chapters/production/multi-cloud)
- [7.4 Testing and deployment](https://learn-inference.com/chapters/production/testing-deployment)
- [7.5 Client code](https://learn-inference.com/chapters/production/client-code)
- [7.6 Where this leaves you](https://learn-inference.com/chapters/production/closing)

### [Glossary](https://learn-inference.com/chapters/glossary)

Every term, defined once

### [Further reading](https://learn-inference.com/chapters/reading)

- [B.1 Architecture](https://learn-inference.com/chapters/reading#architecture)
- [B.2 Developer tools](https://learn-inference.com/chapters/reading#developer-tools)
- [B.3 Frontier open models](https://learn-inference.com/chapters/reading#frontier-models)
- [B.4 GPU infrastructure](https://learn-inference.com/chapters/reading#gpu-infrastructure)
- [B.5 Inference optimization research](https://learn-inference.com/chapters/reading#optimization-research)
- [B.6 Intelligence evaluation](https://learn-inference.com/chapters/reading#evaluation)
