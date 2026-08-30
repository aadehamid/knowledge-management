title: Inference Engineering | Baseten Books
description: Inference Engineering by Philip Kiely is your guide to the hardware, software, techniques, and infrastructure required to run AI models in production.

# Inference Engineering | Baseten Books

## Chapter 0: Inference {#chapter-0-inference}

*Inference Engineering* presents a map of the technologies and techniques that power inference across all three layers of runtime, infrastructure, and tooling.

Chapter 1, Prerequisites, covers the product thinking and AI engineering work that need to be done before inference engineering comes into play: use case definition, latency and cost budgeting, and selecting and evaluating which generative AI models to optimize and deploy.

Chapter 2, Models, introduces the technical architecture of AI models – from large language models to image and video generation models – and establishes where the bottlenecks exist for inference with a special focus on optimizing attention.

Chapter 3, Hardware, starts at the spec sheet for modern GPUs and breaks down compute and memory, then disambiguates architectures and SKUs within NVIDIA’s datacenter-grade offerings before briefly surveying other accelerators on the market.

Chapter 4, Software, builds abstractions from CUDA to frameworks like PyTorch, Transformers, and Diffusers and inference engines like vLLM, SGLang, and TensorRT-LLM. It also introduces Dynamo, NVIDIA’s latest system for large-scale distributed model serving.

Chapter 5, Techniques, discusses key model performance optimization techniques adapted from cutting-edge research and applies them in production: quantization, speculative decoding, KV cache re-use, model parallelism, and disaggregation.

Chapter 6, Modalities, expands inference engineering beyond LLMs to voice and visuals. Many types of generative AI models – vision-language models, embedding models, automatic speech recognition (ASR) models, and speech synthesis models – adapt LLM architectures, meaning inference engineers can run them with the same tools and techniques used with LLMs. Image and video generation models have their own architectures and associated performance optimization techniques.

Chapter 7, Production, concludes the book with a rundown of the important problems to solve in operating infrastructure for and building performant applications on optimized model inference services.

Appendices A and B add a glossary of inference engineering terms and a collection of recommended resources for further reading, respectively.
