[![deepsense.ai](https://deepsense.ai/wp-content/uploads/2024/10/deepsenseai.svg)](https://deepsense.ai/)

* [Tech Expertise](https://deepsense.ai/tech-expertise/)
* Industries
* [Case Studies](/case-studies/)
* AI Insights
* Our Company
* [Join us!](/careers/)

[Contact Us](https://deepsense.ai/contact-us/)

Explore all the technology expertise we have to develop AI solutions

[Explore All](https://deepsense.ai/tech-expertise/)

* [LLMs & RAG](https://deepsense.ai/tech-expertise/llms-rag/)
  + [AI Voice Bots for Enterprise Operations](https://deepsense.ai/tech-expertise/llms-rag/ai-voice-bots-for-enterprise-operations/)
* [Computer Vision](https://deepsense.ai/tech-expertise/computer-vision/)
* [Predictive Analytics](https://deepsense.ai/tech-expertise/predictive-analytics/)
* [MLOps](/tech-expertise/mlops/)
  + [Custom MCP Servers as Part of Enterprise AI Infrastructure](https://deepsense.ai/tech-expertise/mlops/custom-mcp-servers-as-part-of-enterprise-ai-infrastructure/)
* [Edge Solutions](/tech-expertise/edge-solutions/)
* [AI Guidance](https://deepsense.ai/ai-guidance-for-business-and-technical-teams/)

Deploy Agentic RAG Pipelines in Minutes with ragbits

[Get the Code](https://deepsense.ai/rd-hub/ragbits/)

* [Applied AI Blog](/blog/)
* [Open-source](/rd-hub/)
  + [LLM Structured Querying](/rd-hub/db-ally/)
  + [GenAI Development Accelerator](/rd-hub/ragbits/)
  + [3D Gaussian Splatting](https://deepsense.ai/rd-hub/3d-reconstruction/)
  + [GenAI Monitor Framework](https://deepsense.ai/rd-hub/genai-monitor-framework/)
* [Our Resources](/resources/)
  + [Our Interviews with AI Leaders](/resources/our-interviews-with-ai-leaders/)
  + [Our Tech Webinars](/resources/our-tech-webinars/)
  + [Academic Papers](/resources/academic-paper/)
  + [Cookbooks](/resources/cookbook/)

Get to know us, our leadership, development direction, and why we call ourselves applied AI experts.

* [About Us](/about-us/)
* [Our Mission and Values](/our-mission-values/)
* [Our Credentials](/our-credentials/)
  + [Anthropic Service Partner](https://deepsense.ai/enterprise-ready-anthropic-service-partner/)
  + [OpenAI Services Partner](https://deepsense.ai/enterprise-ready-openai-service-partner/)

Look at our open positions and join the applied AI revolution!

[Open Positions](https://deepsense.ai/careers/#find-your-next-role)

* [Careers](/careers/)
* [Why Work at deepsense.ai?](/why-work-at-deepsense-ai/)
* [Applied AI Talent Program](https://deepsense.ai/talent-program/)

With experience across industries,
we deliver impactful projects in these key sectors.

* [Software & Tech](https://deepsense.ai/industry/software-technology/)
* [Pharma](https://deepsense.ai/industry/pharma/)
* [Healthcare](https://deepsense.ai/industry/healthcare/)
* [[Telecoms & Media](https://deepsense.ai/industry/telecoms-media/)](https://deepsense.ai/industry/telecoms-media/)
* [Manufacturing](https://deepsense.ai/industry/manufacturing/)

[Home](https://deepsense.ai/)  [Blog](https://deepsense.ai/blog/)  LLM Inference Optimization: How to Speed Up, Cut Costs, and Scale AI Models

# LLM Inference Optimization: How to Speed Up, Cut Costs, and Scale AI Models

![Katarzyna Rutkowska](https://deepsense.ai/wp-content/smush-webp/2025/01/Katarzyna-Rutkowska-50x50.jpeg.webp)

[Katarzyna Rutkowska](https://deepsense.ai/blog/author/katarzyna-rutkowska/)

9–13 minutes

read

•

15 April, 2025

* [Share using Native toolsShareCopied to clipboard](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/)
* [Click to share on LinkedIn (Opens in new window)LinkedIn](https://www.linkedin.com/sharing/share-offsite/?url=https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/&nb=1)
* [Click to share on X (Opens in new window)X](https://twitter.com/intent/tweet?text=https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/&nb=1)
* [Click to share on Facebook (Opens in new window)Facebook](https://www.facebook.com/sharer/sharer.php?u=https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/&nb=1)

![](https://deepsense.ai/wp-content/smush-webp/2025/04/guillaume-jaillet-Nl-GCtizDHg-unsplash-1024x768.jpg.webp)

Table of contents

1. [Why LLM Inference Optimization Matters](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#tl-dr-llms-for-browser-automation)
2. [Model Distillation: Make LLMs Smaller and Faster](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#model-distillation-make-llms-smaller-and-faster)
   1. [Benefits of Distilling Large Language Models](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#benefits-of-distilling-large-language-models)
   2. [Trade-offs: Speed vs Accuracy in Distilled LLMs](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#trade-offs-speed-vs-accuracy-in-distilled-llms)
3. [Quantization: Reduce Model Size and Inference Cost](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#quantization-reduce-model-size-and-inference-cost)
   1. [How LLM Quantization Works: Reduce Precision, Retain Performance](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#how-llm-quantization-works-reduce-precision-retain-performance)
   2. [Choosing the Right Quantization Strategy for LLMs](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#choosing-the-right-quantization-strategy-for-llms)
   3. [Key Benefits of Quantizing Large Language Models](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#key-benefits-of-quantizing-large-language-models)
   4. [Quantization Challenges and Limitations](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#quantization-challenges-and-limitations)
   5. [Quantization Benchmarks: Memory, Quality & Speed](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#quantization-benchmarks-memory-quality-speed)
      1. [Memory vs Accuracy: Trade-offs in Quantized LLMs](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#memory-vs-accuracy-trade-offs-in-quantized-llms)
      2. [Inference Speed Gains with Quantized Models](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#inference-speed-gains-with-quantized-models)
4. [Continuous Batching: Maximize Throughput in LLM Serving](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#continuous-batching-maximize-throughput-in-llm-serving)
   1. [How Continuous Batching Improves Efficiency](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#how-continuous-batching-improves-efficiency)
   2. [Why Use Continuous Batching for LLM Inference?](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#why-use-continuous-batching-for-llm-inference)
   3. [Batching Trade-offs: Latency vs Throughput](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#batching-trade-offs-latency-vs-throughput)
5. [Key-Value Caching: Speed Up Long-Sequence LLM Generation](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#key-value-caching-speed-up-long-sequence-llm-generation)
   1. [How KV Caching Boosts Inference Speed](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#how-kv-caching-boosts-inference-speed)
   2. [Memory Trade-offs with KV Caching](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#memory-trade-offs-with-kv-caching)
6. [Want to unlock even greater LLM inference performance?](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#want-to-unlock-even-greater-llm-inference-performance)
7. [Summary: LLM Inference Optimization Techniques & Results](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#summary-llm-inference-optimization-techniques-results)
   1. [Distilled Models](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#distilled-models)
   2. [Quantization](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#quantization)
   3. [Continuous Batching](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#continuous-batching)
   4. [KV Cache Optimization](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#kv-cache-optimization)
   5. [How to Combine Optimization Techniques for Best Results](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#how-to-combine-optimization-techniques-for-best-results)
   6. [Thinking about LLM adoption in your company?](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#thinking-about-llm-adoption-in-your-company)
8. [More Resources on LLM Inference Optimization](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#more-resources-on-llm-inference-optimization)

As businesses race to harness the power of Large Language Models, slow responses, rising costs, and hardware demands are becoming major roadblocks. But with the right optimization strategies, it’s possible to unlock faster, leaner, and more scalable LLM performance.

This guide breaks down the key techniques—distillation, quantization, batching, and KV caching—to help you get more out of your models without compromising quality. Let’s get into it.

## Why LLM Inference Optimization Matters

Large Language Models (LLMs) are revolutionizing industries, but optimizing LLM inference remains a challenge due to high latency, cost, and compute demands.

 Slow response times, high computational costs, and scalability bottlenecks can make real-world applications difficult.

This guide covers the top LLM inference optimization strategies – distillation, quantization, batching, and KV caching – to reduce latency, minimize costs, and enhance scalability. We’ll cover:

✔ **Model distillation** – Using a smaller, distilled version of the original model for efficiency.

✔ **Quantization** – Reducing model precision to lower memory usage and improve speed.

✔ **Continuous Batching** – Grouping requests dynamically to maximize throughput.

✔ **KV Caching** – Reducing redundant computation to accelerate token generation.

By implementing these strategies, you can significantly cut costs, reduce latency, and scale LLM applications more effectively. Let’s dive in!

## Model Distillation: Make LLMs Smaller and Faster

One of the most effective ways to optimize LLM inference is **model distillation** – technique where a large, high-accuracy model (the “teacher”) is used to train a smaller, more efficient model (the “student”). This approach retains much of the original model’s knowledge while dramatically improving inference speed and reducing memory requirements.

A great example which we tested in practice is Deepseek-R1 released in different sizes. They come from distillation and allow for more **flexibility** during model deployment – we can choose an appropriate version according to priorities (for example limited hardware). Distillation allowed to compress the original ~1,543 GB model down to ~4GB. It means that we can even deploy the smallest versions on a laptop!

|  |  |  |
| --- | --- | --- |
| Model | Parameters | VRAM Requirement |
| DeepSeek-R1 | 671B | ~1,543 GB |
| DeepSeek-R1-Distill-Llama-70B | 70B | ~181 GB |
| DeepSeek-R1-Distill-Qwen-32B | 32B | ~82 GB |
| DeepSeek-R1-Distill-Qwen-14B | 14B | ~36 GB |
| DeepSeek-R1-Distill-Qwen-7B | 7B | ~18 GB |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | ~3.9 GB |

Source: <https://apxml.com/posts/gpu-requirements-deepseek-r1>

Since the original model is very large, distillation to sizes of 1.5 – 70B is rather aggressive, so it comes with a cost of losing quality.

|  |  |
| --- | --- |
| Model | Quality (MATH-500 pass@1) |
| DeepSeek-R1 | 97.3 |
| DeepSeek-R1-Distill-Llama-70B | 94.5 |
| DeepSeek-R1-Distill-Qwen-32B | 83.9 |

For example on MATH-500 (pass@1) benchmark the original, largest model scores 97.3, 70B version scores 94.5, and the smallest 1.5B version scores 83.9.

For a deeper dive into LLM performance benchmarks, visit [Hugging Face’s DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-R1) model overview and [PromptHub’s](https://www.prompthub.us/blog/deepseek-r-1-model-overview-and-how-it-ranks-against-openais-o1) analysis.

### Benefits of Distilling Large Language Models

✔ **Memory Savings** – Smaller models require less memory, which is a huge plus in constrained resources scenarios.

✔ **Cost** – reduced hardware requirements lowers the costs.

✔ **Faster** **inference** – Smaller models run faster, which is a must if **low latency is a critical requirement**.

### Trade-offs: Speed vs Accuracy in Distilled LLMs

* With smaller size comes an **accuracy loss**, which creates a trade-off that must be carefully chosen.

For many use cases, **distilled models offer the best balance of speed, memory and accuracy**, making them a powerful tool for optimizing LLM inference.

## Quantization: Reduce Model Size and Inference Cost

Why quantize at all? Large Language Models (LLMs) are powerful, but they’re also resource-hungry. Running a full-precision LLM means high GPU memory usage, slow inference, and – most importantly – huge costs. This is where quantization comes to the rescue.

### How LLM Quantization Works: Reduce Precision, Retain Performance

Quantization **reduces the precision** of model weights (e.g., from 32 or 16-bit floating point to 8-bit or even 4-bit integers), significantly **lowering memory usage**, **improving inference speed**, and **cutting down hardware costs** – all while maintaining **acceptable accuracy**. This technique makes it possible to run advanced LLM models on consumer GPUs, edge devices, and cloud environments more **efficiently**.

![](https://lh7-qw.googleusercontent.com/docsz/AD_4nXeo3jtj-3Qn73uWEYDM3AFU_7MfVOa0-BJCsFaWWS8XZqoGpy1PCS64V76IOc9Rv3nIk0i81Y_vF-KBOnTzbiZjnn_FnQxUnP_9ZMjjp5n2t5BZ3IhNx-nREiCNBMot4n7zoZ-pLw?key=Kba1ua3GfN75te6z8yb5Uime)

Source: <https://www.inferless.com/learn/quantization-techniques-demystified-boosting-efficiency-in-large-language-models-llms>

### Choosing the Right Quantization Strategy for LLMs

✔**Post-Training Quantization (PTQ)** – Fast and easy, often quantized checkpoints are already available, **recommended**.

✔  Dynamic Quantization – More accurate than PTQ, applies quantization during inference. In contrast to PTQ, the quantization range is dynamic instead of being fixed.

✔  Quantization-Aware Training (QAT) – Highest quality but requires retraining and **significant resources**.

In practice, PTQ is the most popular and easy to start with. There are available checkpoints on HuggingFace ready to download. Just look at the **quantizations** section:

![](https://lh7-qw.googleusercontent.com/docsz/AD_4nXcQo0p4p-RnhGCZB1z_T1YR2bq9M5YuqUfg1u2VuKa_tXRdejU_66s6MWMMgYtlLQFEFm2zIbgwe-IOukymIfcjYoD7qLh7FPvuxhdm1yJWvONDyfiQMxfWdJ3c2tUtudGIJ1gYZQ?key=Kba1ua3GfN75te6z8yb5Uime)

If you want to dive deeper into quantization methods, check out:

* Alicja Kotyla’s [deep dive into quantization and LoRA](https://deepsense.ai/blog/reducing-the-cost-of-llms-with-quantization-and-efficient-fine-tuning-how-can-businesses-benefit-from-generative-ai-with-limited-hardware/)
* [GPTQ, AWQ, GGUF explanation medium post](https://medium.com/%40mahmoud.bidry11/demystifying-quantization-a-clear-guide-to-understanding-the-concept-and-methods-in-large-language-aa407903da1d#:~:text=It%20does%20save%20a%20lot,75%25%2C%20which%20is%20cool.)
* [LLM quantization](https://noumaan.bearblog.dev/llm-quantization-1/)

### Key Benefits of Quantizing Large Language Models

✔ **Memory Savings** – Quantization significantly reduces model size. This allows LLMs to run on smaller GPUs.

✔ **Cost** – reduced hardware requirements lowers the costs.

✔ **Faster** **inference** – Less memory means faster computation. Quantized models often achieve 2-4× faster inference.

✔ **Quality** – lower memory requirements unlock the potential to fit larger (more powerful) models on the same infrastructure.

✔ **Edge applications** – make LLMs small and fast enough to run on edge devices.

![](https://lh7-qw.googleusercontent.com/docsz/AD_4nXd87KC_Y4TOp1Q-ZMJN6JSnFY0YjLNh0fXIfwpxZ21Z41162Hv30DdBXvriCJsg-edviVe_h-pv4w8GpWWrJIyDX6zLzWnyf9RNBC3H9G_KHtGtITrsCV3BNsr_SVclWJcHGdUkSQ?key=Kba1ua3GfN75te6z8yb5Uime)

Source: <https://www.exxactcorp.com/blog/deep-learning/what-is-quantization-and-llms>

### Quantization Challenges and Limitations

Too good to be true? Quantization is a game-changer for making LLMs more efficient, but it’s not without **trade-offs**. Here’s what you need to consider before jumping in:

* **Accuracy vs. Efficiency** – The lower the precision, the greater the risk of accuracy loss.
* **Hardware Compatibility** – not all GPUs support quantization. If you work with old GPU architectures, chances are your options will be limited. But if your hardware is relatively new – no need to worry!

Yes, quantization comes with trade-offs – but the **benefits far outweigh the drawbacks** for most real-world applications. **8-bit quantization can reduce memory usage by 50%** with **minimal accuracy loss (~1%)**, while **4-bit methods can shrink model size by 75%** while still keeping competitive performance.

In short, quantization isn’t just an optimization – it’s the **key to making LLMs scalable and affordable** in the real world.

### Quantization Benchmarks: Memory, Quality & Speed

If you want to dive deeper and gain more intuition with benchmarks, we will now report two insights:

* Memory (size) vs quality table.
* Speed – before and after quantization comparison.

#### Memory vs Accuracy: Trade-offs in Quantized LLMs

Let’s look at [GGUF](https://huggingface.co/bartowski/DeepSeek-R1-GGUF) reported trade-off for memory vs quality for Deepseek-R1:

![](https://lh7-qw.googleusercontent.com/docsz/AD_4nXffItsEEDyaTPWtuncqn0MKx6OnVYAb2ra1WHWTEPoAq1MYkXjYftuWoFqvBliEnE-Nf3rdcR5mjNtc8hcbT60zIMhNDHK3xS6kmBcXjaZIOX8gRCGc_4Tvc4pyla95EkxK6oOY?key=Kba1ua3GfN75te6z8yb5Uime)

Observations:

* INT8: around **2x** smaller model has **extremely high quality**
* INT4: around **4x** smaller model has **slightly lower quality**
* Below INT4 the memory gain is not that high, and quality drop is drastic. We do not recommend such aggressive quantization.

#### Inference Speed Gains with Quantized Models

At deepsense.ai we benchmarked the generation speed of a few models before and after quantization. Here’s the comparison:

|  |  |  |  |
| --- | --- | --- | --- |
| **Generation speed** | | | |
| Model | Deployment | Base | Quantized – AWQ |
| deepseek (7B) | NVIDIA RTX 4090 (24GB) | 52 tokens/s | 130 tokens/s |
| deepseek (32B) | AWS EC2 g5.12xlarge (96GB) | 22 tokens/s | 50 tokens/s |
| mistral (7B) | AWS EC2 g5.xlarge (24GB) | 28 tokens/s | 88 tokens/s |
| LLama3.3 (70B) | AWS EC2 g5.48xlarge (192GB) | 23 tokens/s | 46 tokens/s |

What we can see from the table is the **massive boost** in the generation speed. Around **2x faster generation for large models**: deepseek (32B) and LLama3.3 (70B), and even more – **2.5-3x speedup for small models**: deepseek (7B) and mistral (7B).

This is real life evidence how much we can gain with AWQ quantization, all that with a minimal (might not be noticeable!) accuracy trade-off.

## Continuous Batching: Maximize Throughput in LLM Serving

Most LLM inference workloads involve multiple users sending requests at different times. Instead of handling each request independently, batching groups requests together. Collective processing increases GPU utilization which increases efficiency. This in turn has a downside – when sequences in a batch have varying lengths, we have to wait for the longest one to finish. This is a bottleneck which can be optimized further with **continuous batching**. Let’s explore how to optimize LLM inference for real-world performance.

### How Continuous Batching Improves Efficiency

Instead of waiting until every sequence in a batch has completed generation, **iteration-level scheduling** is applied, where the batch size is determined per iteration. The result is that once a sequence in a batch has completed generation, a new sequence can be inserted in its place, yielding **higher GPU utilization** than static batching.

![](https://lh7-qw.googleusercontent.com/docsz/AD_4nXfUN0rRgKcan0KPWOhTyJNpJu4Vs2KVq0RRBaST_xdOLZm9E4pmfzvK-gDwC0mk661PgphZXHMglY3oQZa5r67ydBkkKD51-zW2rLT2w0YwWaomcOOKTyaDK5vhKJnZmS2Uh42N?key=Kba1ua3GfN75te6z8yb5Uime)

Source: <https://www.anyscale.com/blog/continuous-batching-llm-inference>

In practice there is a caveat which makes things a bit more complicated: prefill computation and new token generation have different computational patterns. Good news is that LLm serving frameworks such as [vLLM](https://github.com/vllm-project/vllm) already handle this problem.

### Why Use Continuous Batching for LLM Inference?

✔ Allows models to handle hundreds or thousands of concurrent users efficiently.

✔ Increases GPU utilization, leading to higher throughput.

✔ Works well with token streaming, keeping inference fast and responsive.

### Batching Trade-offs: Latency vs Throughput

* Batching makes response times slower for individual requests. If low latency is key – a carefully chosen tradeoff is required.

Even though batching comes with a trade-off, for high-load systems, it’s a must-have optimization to scale efficiently.

## Key-Value Caching: Speed Up Long-Sequence LLM Generation

LLMs generate text **token by token**, and each step requires recalculating attention scores across the entire context. **Key-Value caching** eliminates redundant work by **storing and reusing past attention scores**, speeding up inference. It basically removes the redundancy in token by token computation. This technique is especially useful in long sequence generation, where the speed up will be the highest.

### How KV Caching Boosts Inference Speed

✔ Inference speed up.

✔ Highest benefit is with long sequences.

### Memory Trade-offs with KV Caching

* **KV caching increases memory usage**, since past activations must be stored.

## Want to unlock even greater LLM inference performance?

We highly recommend using [vLLM](https://github.com/vllm-project/vllm) serving framework which supports mentioned optimizations out of the box:

* Continuous batching,
* Quantization,
* KV Cache.

But supports **even more**:

* PagedAttention,
* model execution with CUDA/HIP graph,
* Optimized CUDA kernels,
* Speculative decoding,
* Chunked prefill.

Based on our research and practical experiments on LLM deployments at [deepsense.ai](http://deepsense.ai), we find working with vLLM easy and practical. Vast optimizations available make this framework highly efficient at serving LLMs.

## Summary: LLM Inference Optimization Techniques & Results

### Distilled Models

* **Faster inference** due to reduced parameter count.
* **Lower memory usage**, enabling deployment on smaller hardware.
* **Accuracy loss** – depends on the size of the student model.

### Quantization

* **Inference speed up**: ~2x is realistic.
* **GPU memory reduction**: 2x for int8, 4x for int4.
* Minor quality drop for int8 and int4.
* Aggressive quantization (below int4) affects quality significantly. We do not recommend reducing precision lower than 4bit.

### Continuous Batching

* **Dynamically groups requests** to increase efficiency.
* **Boosts throughput**.
* **Lowers cost** per request.
* Works well with **high-traffic APIs**.
* Trade-off: **latency increase for individual requests**.

### KV Cache Optimization

* **faster inference**, especially for long-text generation.
* **Increases memory usage**.

### **How to Combine Optimization Techniques for Best Results**

Each of these techniques – **distillation, quantization, batching and KV caching** – addresses different inference bottlenecks. By combining them, you can achieve **faster, cheaper, and more scalable LLM deployments** without sacrificing too much accuracy. As AI adoption grows, efficient deployment will be **as important as model performance**.

![](https://deepsense.ai/wp-content/uploads/2024/11/key-visual-blue-7.png)

### Thinking about LLM adoption in your company?

[Contact Us](https://deepsense.ai/contact-us/)

## More Resources on LLM Inference Optimization

1. [vLLM docs](https://docs.vllm.ai/en/latest/)
2. [NVIDIA Inference Optimization post](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
3. [Deepsense.ai blogpost on quantization and LoRA for LLM cost reduction](http://deepsense.ai)

Table of contents

1. [Why LLM Inference Optimization Matters](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#tl-dr-llms-for-browser-automation)
2. [Model Distillation: Make LLMs Smaller and Faster](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#model-distillation-make-llms-smaller-and-faster)
   1. [Benefits of Distilling Large Language Models](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#benefits-of-distilling-large-language-models)
   2. [Trade-offs: Speed vs Accuracy in Distilled LLMs](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#trade-offs-speed-vs-accuracy-in-distilled-llms)
3. [Quantization: Reduce Model Size and Inference Cost](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#quantization-reduce-model-size-and-inference-cost)
   1. [How LLM Quantization Works: Reduce Precision, Retain Performance](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#how-llm-quantization-works-reduce-precision-retain-performance)
   2. [Choosing the Right Quantization Strategy for LLMs](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#choosing-the-right-quantization-strategy-for-llms)
   3. [Key Benefits of Quantizing Large Language Models](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#key-benefits-of-quantizing-large-language-models)
   4. [Quantization Challenges and Limitations](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#quantization-challenges-and-limitations)
   5. [Quantization Benchmarks: Memory, Quality & Speed](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#quantization-benchmarks-memory-quality-speed)
      1. [Memory vs Accuracy: Trade-offs in Quantized LLMs](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#memory-vs-accuracy-trade-offs-in-quantized-llms)
      2. [Inference Speed Gains with Quantized Models](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#inference-speed-gains-with-quantized-models)
4. [Continuous Batching: Maximize Throughput in LLM Serving](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#continuous-batching-maximize-throughput-in-llm-serving)
   1. [How Continuous Batching Improves Efficiency](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#how-continuous-batching-improves-efficiency)
   2. [Why Use Continuous Batching for LLM Inference?](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#why-use-continuous-batching-for-llm-inference)
   3. [Batching Trade-offs: Latency vs Throughput](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#batching-trade-offs-latency-vs-throughput)
5. [Key-Value Caching: Speed Up Long-Sequence LLM Generation](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#key-value-caching-speed-up-long-sequence-llm-generation)
   1. [How KV Caching Boosts Inference Speed](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#how-kv-caching-boosts-inference-speed)
   2. [Memory Trade-offs with KV Caching](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#memory-trade-offs-with-kv-caching)
6. [Want to unlock even greater LLM inference performance?](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#want-to-unlock-even-greater-llm-inference-performance)
7. [Summary: LLM Inference Optimization Techniques & Results](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#summary-llm-inference-optimization-techniques-results)
   1. [Distilled Models](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#distilled-models)
   2. [Quantization](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#quantization)
   3. [Continuous Batching](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#continuous-batching)
   4. [KV Cache Optimization](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#kv-cache-optimization)
   5. [How to Combine Optimization Techniques for Best Results](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#how-to-combine-optimization-techniques-for-best-results)
   6. [Thinking about LLM adoption in your company?](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#thinking-about-llm-adoption-in-your-company)
8. [More Resources on LLM Inference Optimization](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/#more-resources-on-llm-inference-optimization)

[Katarzyna Rutkowska](https://deepsense.ai/blog/author/katarzyna-rutkowska/)

A Senior Machine Learning Engineer with broad experience in machine learning. Focused on natural language processing, LLMs, RAG and document processing. Personally a figure skater and pet lover.

[More resources by this author](https://deepsense.ai/blog/author/katarzyna-rutkowska/)

[![Katarzyna Rutkowska](https://deepsense.ai/wp-content/smush-webp/2025/01/Katarzyna-Rutkowska-150x150.jpeg.webp)](https://deepsense.ai/blog/author/katarzyna-rutkowska/)

Share this post

* [Share using Native toolsShareCopied to clipboard](https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/)
* [Click to share on LinkedIn (Opens in new window)LinkedIn](https://www.linkedin.com/sharing/share-offsite/?url=https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/&nb=1)
* [Click to share on X (Opens in new window)X](https://twitter.com/intent/tweet?text=https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/&nb=1)
* [Click to share on Facebook (Opens in new window)Facebook](https://www.facebook.com/sharer/sharer.php?u=https://deepsense.ai/blog/llm-inference-optimization-how-to-speed-up-cut-costs-and-scale-ai-models/&nb=1)

Posted in

[LLM & RAG](https://deepsense.ai/tags/llm-rag/) [MLOps](https://deepsense.ai/tags/mlops/)

### Explore more insights and resources

* [![Designing Compliance-First Architectures for Regulated Production](https://deepsense.ai/wp-content/smush-webp/2026/05/mcp_session_landing_page_listing-1024x614.png.webp)](https://deepsense.ai/resource/designing-compliance-first-architectures-for-regulated-production/)

  Webinar

  #### [Designing Compliance-First Architectures for Regulated Production](https://deepsense.ai/resource/designing-compliance-first-architectures-for-regulated-production/)
* [![AI Voice Agents & Enterprise Assistants: Lessons from Production](https://deepsense.ai/wp-content/smush-webp/2026/05/Baner-na-strone-1600x960-4-1024x614.png.webp)](https://deepsense.ai/resource/ai-voice-agents-enterprise-assistants-lessons-learned-from-production-deployments/)

  Webinar

  #### [AI Voice Agents & Enterprise Assistants: Lessons from Production](https://deepsense.ai/resource/ai-voice-agents-enterprise-assistants-lessons-learned-from-production-deployments/)
* [![Improving Layout Representation Learning Across Inconsistently Annotated Datasets via Agentic Harmonization](https://deepsense.ai/wp-content/smush-webp/2026/05/academic_paper_vova_unstructured-1024x614.png.webp)](https://deepsense.ai/resource/improving-layout-representation-learning-across-inconsistently-annotated-datasets-via-agentic-harmonization/)

  Academic paper

  #### [Improving Layout Representation Learning Across Inconsistently Annotated Datasets via Agentic Harmonization](https://deepsense.ai/resource/improving-layout-representation-learning-across-inconsistently-annotated-datasets-via-agentic-harmonization/)

![](https://deepsense.ai/wp-content/uploads/2024/10/6-a-2.png)

Transform your business
with AI solutions

[Let’s talk](https://deepsense.ai/contact-us/)

![](https://deepsense.ai/wp-content/uploads/2024/10/6-a-3.png)

![](https://deepsense.ai/wp-content/uploads/2024/10/6-a-4.png)

[![deepsense.ai](https://deepsense.ai/wp-content/uploads/2024/10/deepsenseai.svg)](https://deepsense.ai/)

* [LinkedIn](https://www.linkedin.com/company/deepsense-ai/)
* [Facebook](https://www.facebook.com/deepsenseai)
* [X](https://twitter.com/deepsense_ai)
* [YouTube](https://www.youtube.com/%40deepsenseai)
* [Medium](https://medium.com/deepsense-ai)

* [Tech Expertise](https://deepsense.ai/tech-expertise/)

* [LLMs & RAG](https://deepsense.ai/tech-expertise/llms-rag/)
* [MLOps](https://deepsense.ai/tech-expertise/mlops/)
* [Computer Vision](https://deepsense.ai/tech-expertise/computer-vision/)
* [Edge Solutions](https://deepsense.ai/tech-expertise/edge-solutions/)
* [Predictive Analytics](https://deepsense.ai/tech-expertise/predictive-analytics/)

* Industries

* [Software & Technology](https://deepsense.ai/industry/software-technology/)
* [Manufacturing](https://deepsense.ai/industry/manufacturing/)
* [Pharma](https://deepsense.ai/industry/pharma/)
* [Healthcare](https://deepsense.ai/industry/healthcare/)
* [Telecoms & Media](https://deepsense.ai/industry/telecoms-media/)

* [Case studies](https://deepsense.ai/case-studies/)

* [Nielsen](https://deepsense.ai/case-studies/ai-powered-image-recognition-for-faster-fmcg-insights/)
* [AdaCore](https://deepsense.ai/case-studies/ai-copilots-impact-on-productivity-in-revolutionizing-ada-language-development/)
* [NOAA](https://deepsense.ai/case-studies/recognizing-wildlife-in-aerial-imagery-with-98-reduction-in-image-analysis-time/)
* [Volkswagen](https://deepsense.ai/case-studies/reinforcement-learning-speeds-up-autonomous-driving/)
* [More case studies](/case-studies/)

* Insights

* [R&D Hub](https://deepsense.ai/rd-hub/)
* [db-ally](https://deepsense.ai/rd-hub/db-ally/)
* [3D Gaussian Splatting](https://deepsense.ai/rd-hub/3d-reconstruction/)

* [Blog](/blog/)

* Company

* [Our Mission and Values](https://deepsense.ai/our-mission-values/)
* [Why Work at deepsense.ai](https://deepsense.ai/why-work-at-deepsense-ai/)
* [Our Credentials](https://deepsense.ai/our-credentials/)
* [Careers](https://deepsense.ai/careers/)
* [Summer Internship Program](https://deepsense.ai/intership/)

Our Main Offices

Palo Alto

2100 Geng Road, Suite 210
Palo Alto, CA 94303
United States of America

Warszawa HQ

al. Jerozolimskie 44
00-024 Warsaw
Poland

© 2025 deepsense.ai All rights reserved.

* [Privacy Policy](/privacy-policy/)
* [Terms of Service](https://deepsense.ai/terms-of-service/)
* [Code of Ethics](https://deepsense.ai/code-of-ethics/)