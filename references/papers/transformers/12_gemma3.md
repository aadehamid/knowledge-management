title: LLMs-from-scratch/ch05/12\_gemma3 at main · rasbt/LLMs-from-scratch
description: Implement a ChatGPT-like LLM in PyTorch from scratch, step by step - LLMs-from-scratch/ch05/12\_gemma3 at main · rasbt/LLMs-from-scratch

# LLMs-from-scratch/ch05/12\_gemma3 at main · rasbt/LLMs-from-scratch

This [standalone-gemma3.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/12_gemma3/standalone-gemma3.ipynb) Jupyter notebook in this folder contains a from-scratch implementation of Gemma 3 270M. It requires about 2 GB of RAM to run.

The alternative [standalone-gemma3-plus-kvcache.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/12_gemma3/standalone-gemma3-plus-kvcache.ipynb) notebook adds a KV cache for better runtime performance (but adds more code complexity). To learn more about KV caching, see my [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) article.

| Model | Mode | Hardware | Tokens/sec | GPU Memory (VRAM) |
|----|----|----|----|----|
| Gemma3Model 270M | Regular | Mac Mini M4 CPU | 8 | - |
| Gemma3Model 270M | Regular compiled | Mac Mini M4 CPU | 9 | - |
| Gemma3Model 270M | KV cache | Mac Mini M4 CPU | 130 | - |
| Gemma3Model 270M | KV cache compiled | Mac Mini M4 CPU | 224 | - |
|  |  |  |  |  |
| Gemma3Model 270M | Regular | Mac Mini M4 GPU | 16 | - |
| Gemma3Model 270M | Regular compiled | Mac Mini M4 GPU | Error | - |
| Gemma3Model 270M | KV cache | Mac Mini M4 GPU | 23 | - |
| Gemma3Model 270M | KV cache compiled | Mac Mini M4 GPU | Error | - |
|  |  |  |  |  |
| Gemma3Model 270M | Regular | Nvidia A100 GPU | 28 | 1.84 GB |
| Gemma3Model 270M | Regular compiled | Nvidia A100 GPU | 128 | 2.12 GB |
| Gemma3Model 270M | KV cache | Nvidia A100 GPU | 26 | 1.77 GB |
| Gemma3Model 270M | KV cache compiled | Nvidia A100 GPU | 99 | 2.12 GB |

Below is a side-by-side comparison with Qwen3 0.6B as a reference model; if you are interested in the Qwen3 0.6B standalone notebook, you can find it [here](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3).

[![](https://camo.githubusercontent.com/c89c2877e8943617dc2448c560f69a5ce47f5557076871da3d45b9058c3e5a42/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f626f6e75732f67656d6d61332f67656d6d61332d76732d7177656e332e77656270)](https://camo.githubusercontent.com/c89c2877e8943617dc2448c560f69a5ce47f5557076871da3d45b9058c3e5a42/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f626f6e75732f67656d6d61332f67656d6d61332d76732d7177656e332e77656270)

To learn more about the architecture differences and read about comparisons with other architectures, see my [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) article.
