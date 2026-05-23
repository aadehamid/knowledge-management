[Hippocratic AI + Modular to power real-time patient conversations. Read More →](https://modular.com/blog/hippocratic-ai-partners-with-modular-to-power-flexible-high-quality-inference-for-real-time-patient-conversations)

Back

* Product
* Solutions
* Resources
* Open Source
* [Docs](https://docs.modular.com/max/)
* [Blog](/blog)
* Company

[Request a Demo](/request-demo)[Sign up](https://console.modular.com/signup?utm_source=topNav)

* + Inference Products

    - [Shared Endpoints

      Access frontier models via an API](/inference/shared-endpoints)
    - [Dedicated Endpoints

      Mission critical reliability](/inference/dedicated-endpoints)
    - [Custom models

      Your model, peak performance](/inference/custom-models)
  + Deployment Options

    - [Our Cloud

      Fully managed, pay by usage](/deploy/our-cloud)
    - [Your Cloud

      Modular stack in your VPC](/deploy/your-cloud)
    - [Pricing

      Flexible plans for every team](/pricing)
  + Models

    [![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/69f1f304e32c32ad1468e64f_deepseek.webp)](/models/deepseek-v4-pro)

    DeepSeek V4 Pro

    [![FLUX.2 Klein 9B logo](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/69c3a896a235a8dc5cbee055_m5YoF33abJ09vcwFxt1Mj.webp)](/models/flux2-klein)

    FLUX.2 Klein 9B

    [![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/69f1f2f58b50d28745a8f67b_glm.webp)](/models/glm-5-1)

    GLM-5.1

    [![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/69f1e91cd64ccdb839384b54_kimik26.webp)](/models/kimi-k2-6)

    Kimi K2.6

    [![Wan 2.2 T2V A14B logo](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/69bafcab62c54f8a76122234_-s1gyJfvbE1RgO5iBeNOi.webp)](/models/wan2-2-t2v-a14b-diffusers)

    Wan 2.2 T2V A14B

    [![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/69bd89feec1cad6fa2305777_Brain.svg)](/models)

    View All
* + - [Text to audio

      Turn text into natural speech](/solutions/audio)
    - [Image generation

      Generate images from text prompts](/solutions/image-generation)
    - [Code generation

      Generate production-ready code](/solutions/code-generation)
    - [Video generation

      Generate videos from text+image](/solutions/video-generation)
  + - [Agentic

      Deploy AI agents anywhere](/solutions/agentic)
    - [Custom Models

      Kernel-level model control](/inference/custom-models)
    - [Case Studies

      Proven results from real customers](/customers)
  + Models

    [![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/69f1f304e32c32ad1468e64f_deepseek.webp)](/models/deepseek-v4-pro)

    DeepSeek V4 Pro

    [![FLUX.2 Klein 9B logo](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/69c3a896a235a8dc5cbee055_m5YoF33abJ09vcwFxt1Mj.webp)](/models/flux2-klein)

    FLUX.2 Klein 9B

    [![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/69f1f2f58b50d28745a8f67b_glm.webp)](/models/glm-5-1)

    GLM-5.1

    [![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/69f1e91cd64ccdb839384b54_kimik26.webp)](/models/kimi-k2-6)

    Kimi K2.6

    [![Wan 2.2 T2V A14B logo](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/69bafcab62c54f8a76122234_-s1gyJfvbE1RgO5iBeNOi.webp)](/models/wan2-2-t2v-a14b-diffusers)

    Wan 2.2 T2V A14B

    [![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/69bd89feec1cad6fa2305777_Brain.svg)](/models)

    View All
* + - [MAX Framework

      GenAI native modeling & serving

      ![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/690d3398ec2956898ed94e68_max-hero.png)](/open-source/max)
    - [Mojo Language

      The best GPU & CPU performance](https://mojolang.org/)
    - [Self-Hosted

      MAX+Mojo self-hosted by you](/open-source/self-hosted)
    - [Community

      Build the future of AI together](/open-source/community)
    - [Mojo Agent Skills

      Official AI agent skills from Modular](https://github.com/modular/skills)
* + [Docs

    Deploy GenAI models, our cloud or yours](https://docs.modular.com/)
  + [Model Library

    Latest supported open models](/models)
  + [Mojo Docs

    Write high-performance kernels for CPUs and GPUs](https://mojolang.org/docs/)
  + [Community

    Build the future of AI together](/open-source/community)
* + [About

    Build AI for anyone, anywhere.](/company/about)
  + [Careers

    👋  We’re currently hiring!](/company/careers)
  + [Culture

    What we believe](/company/culture)
  + [Contact Us

    Request a demo](/request-demo)

[Request a Demo](/request-demo)[Sign up](https://console.modular.com/signup?utm_source=topNav)

close

February 5, 2026

# The Five Eras of KVCache

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6984f8f1b33180e5936a7b88_brian.jpeg)

Brian Zhang

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6984f903eb465bf32916cf6d_Gemini_Generated_Image_egiucbegiucbegiu.jpeg)

Engineering

## Introduction

**Key–Value Cache (KV cache/KVCache)** is a foundational building block of modern LLM serving systems. It stores past attention states so the model can generate new tokens efficiently without excessive re-computation.

There are two phases to LLM inference: **Prefill** and **Decode**. In the Prefill phase, the attention states are computed for each token in the input prompt. In the subsequent Decode phase, new tokens are generated one by one in an autoregressive fashion by attending on the Key-Value associated with previous tokens.

![https://www.nature.com/articles/s41586-023-06647-8](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/6984eca926d742f1e540fb6d_autoregressive-downarrows-20260128123257.svg)

<https://www.nature.com/articles/s41586-023-06647-8>

vLLM, SGLang, TensorRT-LLM, and MAX Serve are all built on top of increasingly sophisticated KV cache management. This blog explores the evolution and role of the KV cache in these inference engines

## Era 0: Pre-GenAI (<2017)

Before transformers took over, deep learning was dominated by *stateless, feed-forward* architectures like ResNet, YOLO, VGG, and Inception. These models did not require persistent state across inference steps, so the concept of a KVCache simply didn’t exist even in inference frameworks like ONNX or TensorRT.

## Era 1: Continuous KV Cache (2017)

The original [transformer (2017)](https://arxiv.org/pdf/1706.03762) established the architecture that would eventually dominate ML. This design was a departure from prior models, requiring a KVCache to efficiently keep track of the state associated with each request. Nevertheless, the major step-change in intelligence enabled by transformers more than justified their added complexity.

At the time, early LLM serving engines implemented KV caches naively:

* For each request, they preallocated a contiguous KV tensor with `max_seq_len` tokens.
* The storage was `2 x num_layers × num_heads × head_dim × max_seq_len` per request.

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/6984eca93733ef1fa4c63095_continuous_kvcache_%25281%2529.svg)

This Contiguous KV cache design was extremely wasteful, but still offered huge performance gains over recomputing attention keys/values for each token:

* ✔ Simple
* ✘ Memory usage scales aggressively due to the `max_seq_len × batch_size` factor
* ✘ Constrained `max_batch_size` due to limited memory capacity
* ✘ High memory fragmentation due to variable-length requests
* ✘ Most request are far shorter than `max_seq_len`, leaving much wasted capacity

This was the approach of early inference engines like HuggingFace Transformers.

## Era 2: PagedAttention (2023)

A breakthrough arrived with [PagedAttention](https://arxiv.org/abs/2309.06180), introduced by **vLLM**. The key idea was to borrow [a technique from Operating Systems](https://en.wikipedia.org/wiki/Page_table) by allocating KV in fixed-size pages that could be dynamically allocated as sequences grew.

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/6984eca9d38ab8fb3945cbba_paged_kvcache.svg)

**Benefits:**

* ✔ Dramatically improves memory utilization and reduces fragmentation
* ✔ Enables hundreds / thousands of concurrent requests
* ✔ Drives up throughput via larger batch sizes
* ✔ Allows for efficient KV cache reuse via [Prefix Caching](https://arxiv.org/pdf/2312.07104), a huge throughput multiplier for multi-turn chat workloads

PagedAttention became the de-facto standard for LLM serving, leading to new inference engines like TensorRT-LLM and SGLang.

## Era 3: Heterogenous KV Caches (2024)

The world of ML and the LLM serving landscape is far more complex now. New optimizations along with modern multimodal and hybrid models require *multiple different kinds of state*, each with separate caching requirements. In this Era, the term “KV Cache” is being stretched far beyond its original meaning.

1. **Speculative decoding** accelerates LLM inference by having a small draft model generate multiple tokens ahead and then using a larger target model to verify and accept those tokens in a single pass. With this technique, a separate KV cache needs to be maintained for the draft and target model.
2. [**Vision encoders**](https://github.com/vllm-project/vllm/pull/9871) in Vision–Language Models (VLMs) generate large image embeddings that can be cached and reused across requests. While this differs from the traditional notion of a “KV cache” or prefix caching, it follows the same underlying principle of memoizing expensive intermediate states. Models which benefit from this cache include QwenVL and InternVL.
3. **Quantized KV Cache**: Low precision datatypes like FP8 help reduce the storage requirements of the KV cache and rely on per-tensor/row/block scaling factors to preserve numerical range. This requires the KV cache implementation to also manage memory for these scaling factors.
4. **Sliding Window Attention (SWA)** limits each token to attend only to the preceding `window_size` tokens instead of the entire sequence, reducing memory and compute. As a result, KV cache management and prefix caching must track which tokens fall within the current window, making cache hits and evictions more complex than in full attention.![Fig 11. https://arxiv.org/pdf/2503.18292](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/6984ecaaa581b58f7daac003_image.png)

   Fig 11. <https://arxiv.org/pdf/2503.18292>
5. [**Mamba / State Space Models**](https://arxiv.org/pdf/2312.00752) replaces attention with a recurrent state that updates a single large vector for each new token. This makes prefix caching more complex because serving systems must decide when and how to checkpoint or store the evolving state vector for future reuse.
6. **Composite Models** are composed of multiple sub-models. For example, it is a common pattern to combine an LLM backbone with an audio decoder. Each of these sub-models may require maintaining separate KV caches.
7. **Hybrid Models** combine multiple layer types within a single model, which often necessitates maintaining multiple KV caches to handle each layer’s distinct attention or state mechanism. Examples include:
   1. Sliding Window Attention + Full Attention (Gemma2/3, Ministral, GPT-OSS, Cohere)
   2. Mamba + Full Attention (Jamba, Bamba, Minimax)
   3. Local Chunked + Full Attention (Llama4)![Fig 1. https://arxiv.org/pdf/2503.18292](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/6984ecaae42b0afd0dea3c60_image.png)

   Fig 1. <https://arxiv.org/pdf/2503.18292>

- This list is non-exhaustive. There are a ton of other ideas like [Transfusion](https://arxiv.org/pdf/2408.11039v1) for joint text–image generation, [dynamic KV cache compression](https://developer.nvidia.com/blog/dynamic-memory-compression/), [Cross-Attention](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) (not to be confused with [Cross-Layer Attention](https://arxiv.org/pdf/2405.12981)), etc.

This heterogeneity and diversity of KV cache’s with different shapes, lifetimes, and properties led to the creation of specialized managers in modern LLM serving engines. For example, vLLM has the [Vision Encoding Cache](https://github.com/vllm-project/vllm/pull/9871), [Mamba Cache](https://github.com/ZZBoom/vllm/blob/c7d93c1a9605767733d9e091d98e4ee5df747565/vllm/model_executor/models/mamba_cache.py), etc, in additional to its normal KV cache.

There are several **challenges** emerging with this design:

* ✘ Memory fragmentation due to multiple KV cache managers can lead to small batch sizes
* ✘ Challenging to predict at server startup how much memory to allocate per KV cache
* ✘ Disjoint Prefix Caching implementations lead to suboptimal cache hit rates
* ✘ Diversity makes feature composition challenging

## Era 4: Distributed KV Cache (2025+)

As LLMs grow in size and handle increasing workloads, a single GPU or node becomes insufficient. Now LLM serving and the KV cache is becoming multi-node and distributed, often spanning an entire datacenter. Managing the massive scale of the KV cache requires new techniques as such:

* [**Disaggregated Inference:**](https://arxiv.org/pdf/2401.09670) LLM inference is divided into **Prefill** and **Decode** phases, deployed and scaled on separate model instances to reduce interference and optimize resource usage. A key challenge is efficiently transferring the KV cache from Prefill nodes to Decode nodes. Recently new variants of disaggregation have emerged like [Encoder Disaggregation](https://blog.vllm.ai/2025/12/15/vllm-epd.html).
* [KV Cache-aware Load Balancing:](https://docs.nvidia.com/dynamo/archive/0.4.0/architecture/kv_cache_routing.html) Request routing prioritizes instances that already hold the relevant KV cache, maximizing **prefix cache hits**. This requires a cluster-wide view of the current state of the KV cache on each of the individual instances.
* [**Hierarchical KVCache:**](https://arxiv.org/pdf/2407.00079v1) To increase KV cache capacity, **cold pages** can be spilled from GPU memory to more abundant CPU RAM or SSD. This extends the effective KV cache size while keeping the hot, frequently accessed pages in GPU memory for low-latency access. The higher latency of loading/storing of KV cache for one model layer from a lower tier of the cache can be hidden by overlapping it with the GPU execution for the prior layer.

Many new kubernetes-native inference solutions like Nvidia Dynamo, vLLM Production Stack, llm-d, or AIBrix have emerged to tame this complexity. However, distributed LLM inference is still very hard:

* ✘ Many existing optimizations or architectures are still incompatible with distributed inference like speculative decoding or VLMs
* ✘ Despite the wide availability of open-source solutions, it still requires expert knowledge and a lot of patience to deploy
* ✘ Inter-node GPU networking over Infiniband or RoCE is challenging and many libraries like [NIXL](https://github.com/ai-dynamo/nixl) are nascent
* ✘ There are many inherent problems for large-scale distributed systems such as managing failover, stragglers, hardware defects, auto-scaling, etc

## Era 5: Unified Hybrid KV Caches (2025+)

The next stage is building **unified KV memory systems** where *many heterogeneous KV types share a common memory pool* rather than isolated allocators. Another overarching theme in this era is striving for **composability between all available optimizations.**

This evolution is happening today!

**Emerging approaches:**

1. [vLLM / Jenga – Huge Pages + LCM Sizing](https://arxiv.org/pdf/2503.18292)
   1. Use **huge pages** with sizes chosen as the **least common multiple** of smaller page formats so different KV shapes can co-exist efficiently.
   2. Unified Prefix Caching design that takes into consideration many KV caches at once to improve balance and hit rate.

![Jenga](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/698601a72e69ae512df97049_image.png)

Jenga

![Jenga](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/698601a745c32f01551549c3_image.png)

Jenga

2. [SGLang – CUDA Virtual Memory](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)

1. SGLang uses **CUDA Virtual Memory APIs** to dynamically remap device memory and unify different KV regions.
2. his enables virtually contiguous but physically scattered KV pages

![https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/6984ecaa3733ef1fa4c63150_image.png)

<https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/>

1. Significant effort is also being invested into feature composability. In fact, this is one of the critical tenets of the [2025Q4 SGLang roadmap](https://github.com/sgl-project/sglang/issues/12780). For instance, one should be able to run a VLM model with Speculative Decoding across multiple nodes in a disaggregated setup. This will require long-term software investment and re-architecting core components of the inference engine.

# Conclusion

What began as a simple optimization—caching attention states to avoid recomputation—has evolved into one of the most complex subsystems in modern AI infrastructure. Each era has brought new challenges: memory fragmentation, heterogeneous model architectures, distributed coordination, and now the need for unified systems that compose cleanly across all these dimensions. As new models, optimizations, and hardware emerge, KV cache management will require innovation across all layers of the LLM inference stack from GPU kernels to cluster-scheduling.

This complexity is precisely why we built [MAX](https://www.modular.com/max) with a ground-up approach to KV cache management. Combined with [Mojo's](https://www.modular.com/mojo) performance and flexibility, we're building infrastructure that handles today's models while adapting to tomorrow's innovations.

Interested in how MAX handles KV cache for your workloads? [Get started here](https://docs.modular.com/max/get-started/) or [join our community](https://www.modular.com/community) to discuss with the team.

## Read more from Modular

[View all blogs](/blog)

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6a0f31d3aca208cbec6e0a9a_option01.jpeg)

Why LLM Inference Needs a New Kind of Router - Part 2

May 21, 2026

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/69fe32cfa1d68466e6170408_MAX-inference-router.jpeg)

Why LLM Inference Needs a New Kind of Router - Part 1

May 8, 2026

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/69e27c33262bc00a7a27ed1c_TileTensorCover.jpeg)

TileTensor Part 1 - Safer, More Efficient GPU Kernels

April 13, 2026

Build the future of AI with Modular

[Get started - FREE](https://console.modular.com/signup)

[View Editions](/pricing)

[](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f%2F68cb292399824cd6e809145c_bg-loop-transcode.mp4)

* ![Person with blonde hair using a laptop with an Apple logo.](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/68cc733ff9050921bab7782c_emoji-dev.png)

  Sign up today

  Signup to our Cloud Platform today to get started easily.

  [Sign Up](https://docs.modular.com/max/get-started)
* ![Magnifying glass emoji with black handle and round clear lens.](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/68cc733fb111718bbd49ca31_emoji-zoom.png)

  Browse open models

  Browse our model catalog, or deploy your own custom model

  [Browse models](https://www.modular.com/models)

## Sign up for our newsletter

Get all our latest news, announcements and updates delivered directly to your inbox. Unsubscribe at anytime.

⚠️ This form requires JavaScript to function. Please enable JavaScript in your browser to continue.

Email\*

First Name

Last Name

Thanks for signing up to our newsletter! 🚀

Thank you,

Modular Sales Team

Oops! Something went wrong while submitting the form.

Latest from our blog:

[![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6a0b4487a1cc9e7f872840da_HipCaseStudy.png)

Hippocratic AI partners with Modular to power flexible, high-quality inference for real-time patient conversations](/blog/hippocratic-ai-partners-with-modular-to-power-flexible-high-quality-inference-for-real-time-patient-conversations)

Get the latest news,

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a81f/68cc76bcf732ee52e8efb9d9_icon-email.svg)

Join our Newsletter

* INFERENCE

  + [Shared Endpoints](/inference/shared-endpoints)
  + [Dedicated Endpoints](/inference/dedicated-endpoints)
  + [Custom Models](/inference/custom-models)
* DEPLOYMENTS

  + [Our Cloud](/deploy/our-cloud)
  + [Your Cloud](/deploy/your-cloud)
  + [Pricing](/pricing)
* Solutions

  [Text to audio](/solutions/audio)

  [Image generation](/solutions/image-generation)

  [Code generation](/solutions/code-generation)

  [AI Agents](/solutions/agentic)
* Open SOURCE

  + [MAX](/open-source/max)
  + [Mojo🔥](/dev/mojo)
  + [Agent Skills](https://github.com/modular/skills)
* Connect

  + [Blog](/blog)
  + [Community](/open-source/community)
  + [Report a security issue](/company/report-issue)
* Company

  + [About Us](/company/about)
  + [Culture](/company/culture)
  + [Careers](/company/careers)
  + [Request a demo](/request-demo)

Copyright © 2026 Modular Inc

[Terms](/legal/terms), [Privacy](/legal/privacy) & [Acceptable Use](/legal/aup)

Join our newsletter

Get all our latest news, announcements and updates delivered directly to your inbox. Unsubscribe at anytime.

Email\*

First Name

Last Name

Thanks for signing up to our newsletter! 🚀

Thank you,

Modular Sales Team

Oops! Something went wrong while submitting the form.

Return to page

Modular Champions

X/X

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c417ad2de85aa7c73eed_6923596dc4d3dc4a769cfda4_owen-hilyard.jpeg)

Owen Hilyard

I'm a PhD student at the University of New Hampshire in the Cloud Computing Lab, where I conduct research on hardware acceleration of networking and distributed systems reliability. I'm also a former component maintainer for DPDK, the Data Plane Development Kit, which is where I got my start looking "under the hood" at networking before resigning to work in my PhD. All of this means I like making computers go fast, and Mojo + MAX is a great place to combine my love of hardware, high performance software, and my interest in programming languages. I also act as one of the community moderators for Modular on the Discord server and Discourse forum.

Connect with Owen Hilyard

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c4172ce608abed214612_69235996eaf244c8c1584940_seth-stadick.jpeg)![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c4172ce608abed21460c_6924a25d0e1bcaded722ecb7_emoji-explode.png)

Seth Stadick

I'm a Bioinformatics Software Engineer passionate about building high-performance systems. For the past six years, I've developed in Rust across environments ranging from Raspberry Pi devices to full-scale HPC clusters in the cloud. I'm excited about Mojo's potential to reshape how we approach performance-critical computing.When I'm not coding, I’m usually spending time with my family — or trying to land new tricks on a skateboard (I just learned to Ollie)!

Connect with Seth Stadick

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c4177507f06c6a92ec1b_692359bc01147a2f4c3cfd1a_brian-grenier.jpeg)![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c4177507f06c6a92ec17_6924a269769e4a94d866aa27_emoji-machevuoi.png)

Brian Grenier

I am C++ developer working for a cardiac image processing company. I've been a Mojo standard library contributor since 2024. I also actively maintain two libraries, EmberJSON, a JSON library written in pure Mojo, and Kelvin, a type safe dimensional analysis library. I often hang out in the Mojo discord and forum!

Connect with Brian Grenier

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c417d0370bc41829c918_69235a2d17b125834ba5d721_martin-vuyk.jpeg)![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c417d0370bc41829c915_6924a274cb0a7c1a1952d336_emoji-q2.png)

Martin Vuyk

I'm a Mechatronics Engineer who pivoted into Software Development. I like tackling complex problems and building lasting solutions. I invest my time into things that I think will be impactful.

Connect with Martin Vuyk

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c4178e2aa2dbe4926c97_692359d23c0eb0b468eb85f7_sawyer-bergeron.jpeg)![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c4178e2aa2dbe4926c9c_6924a281d231594a12889d97_emoji-q.png)

Sawyer Bergeron

Compiler, PL, and systems/performance engineering enthusiast with an eyebrow-raising amount of VAX assembly experience. Collector (of hobbies). Enjoys bagels just a bit too much.

Connect with Sawyer Bergeron

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c4184404f567d86c3fa0_692359d8e862996d6e873fd0_valentin-erokhin.jpeg)![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c4184404f567d86c3faf_6924a2a54f7f4a53bc9d3e84_emoji-bee.png)

Valentin Erokhin

Author of Lightbug, a Mojo HTTP Framework (https://github.com/Lightbug-HQ/lightbug\_http)

Connect with Valentin Erokhin

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c418a0e1f9e5c50ed4ed_692359dc98d5de59b8ba0c44_sora.jpeg)![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c418a0e1f9e5c50ed4ea_6924a2b1c56b7843eca61fbf_emoji-upsidedown.png)

Sora

Connect with Sora

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c417cd73c8fb08635a25_692359c79405151086edae43_maxim-zaks.jpeg)![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c417cd73c8fb08635a2a_6924a2bfbcf7b7f70604b3d4_emoji-catscare.png)

Maxim Zaks

I tell computers how to waste electricity, hopefully in an efficient or at least useful way.

Connect with Maxim Zaks

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c4175a760300836082b5_6924a60ada600939ad5c5f18_IMG_20210929_110905618_HDR%2520-%2520Gabriel%2520de%2520Marmiesse.jpeg)![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c4175a760300836082b1_6924a691ca8d9f4a839c3826_emoji-cat.png)

Gabriel de Marmiesse

Former core dev of Keras, author of Python-on-whales, currently working at Kyutai to democratize AI through open science

Connect with Gabriel de Marmiesse

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6977d5022d2e3029ab1dbec8_MaxB-Profile.jpg)![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c417da688b9ef16432e5_6924a2d7e11925ae81cd9348_emoji-sunflower.png)

Max Brylski

Explorer of the computational universe

Connect with Max Brylski

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c418ca0bcd2d54f4a6fb_69262f1707de6d827b446673_avatar-max.png)![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/6926c418ca0bcd2d54f4a6f8_6924a2cf819ab6c2525890bf_emoji-monkey.png)

Tilli Fe

Connect with Tilli Fe

Return to page

Leadership team

X/X

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/68c9c3107effc2ea46e1b279_Chris.jpg)

Chris Lattner

[Distinguished Leader](http://www.nondot.org/sabre/) who founded and scaled critical infrastructure including [LLVM](https://llvm.org/), [Clang](https://clang.llvm.org/), [MLIR](https://mlir.llvm.org/), [Cloud TPUs](https://cloud.google.com/tpu) and the [Swift](https://swift.org/) programming language. Chris built AI and core systems at multiple world leading technology companies including Apple, Google, SiFive and Tesla.

Connect with Chris Lattner

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/68c9c3107effc2ea46e1b267_Tim.jpg)

Tim Davis

[Repeat Entrepreneur](http://www.timdavis.com/about) and Product Leader. Tim helped build, found and scale large parts of Google's AI infrastructure at [Google Brain](https://research.google/teams/brain/) and Core Systems from APIs ([TensorFlow](http://www.tensorflow.org)), Compilers ([XLA](https://www.tensorflow.org/xla) & [MLIR](https://www.blog.google/technology/ai/mlir-accelerating-ai-open-source-infrastructure/)) and runtimes for server (CPU/GPU/TPU) and [TFLite](https://www.youtube.com/watch?v=Jjm7MT6W0Dc) (Mobile/Micro/Web), [Android ML](https://developers.google.com/ml-kit) & [NNAPI](https://source.android.com/devices/architecture/modular-system/nnapi), large model infrastructure & OSS for billions of users and devices. Loves running, building and scaling products to [help people](https://www.youtube.com/watch?v=o623TB-mY6A), and [the world](https://www.youtube.com/watch?v=UT2noVDFoaA).

Connect with Tim Davis

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/68c9c3107effc2ea46e1b26c_Mostafa.jpg)

Mostafa Hagog

Mostafa is a seasoned engineering leader in high-performance computing. During his tenure at [NVIDIA](https://www.nvidia.com/en-us/), he served as Engineering Director and led teams to develop optimized deep learning libraries like cuDNN and CUTLASS, revolutionizing GPU-accelerated AI. At [SiFive](https://www.sifive.com/), as VP of Software, Mostafa assumed a leadership role guiding teams in the development of an MLIR/LLVM-based software stack for SiFive Intelligence & performance cores. His contributions also extend to optimizing [Intel](https://www.intel.com/content/www/us/en/homepage.html) GPU hardware/software features, playing a pivotal role in developing the AVX1/2 SIMD ISA for Intel CPUs, and contributing to the GNU C Compiler. Mostafa holds a Master of Science in Electrical Engineering from the [Technion](https://www.technion.ac.il/en/home-2/), with a specialization in compiler optimizations. His unwavering passion for innovation continues to drive advancements in the field of high-performance computing.

Connect with Mostafa Hagog

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/68c9c3107effc2ea46e1b26b_Kalor.jpg)

Kalor Lewis

Kalor is Modular's VP, Finance and leads all our Finance operations. Prior to Modular, Kalor was a VP, Finance at [Fivetran](https://fivetran.com/) where he was the first finance hire in 2018 and built out the companies entire finance function. Before Fivetran, Kalor was part of [Palantir Technologies](https://www.palantir.com/), where he scaled their strategic finance function.

Connect with Kalor Lewis

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/68c9c3107effc2ea46e1b268_Eric.jpg)

Eric Johnson

Product leader who has built and scaled AI applications and infrastructure. Eric led the TensorFlow API, Compiler, and Runtime teams at [Google Brain](https://research.google/teams/brain/) and Core Systems, including the founding of [TFRT](https://blog.tensorflow.org/2022/02/tfrt-progress-update.html) and the productionization of [JAX](https://github.com/google/jax). He holds an MBA from [Wharton](https://www.wharton.upenn.edu/) and Computer Science MS from Penn and loves soccer, fitness, and the great outdoors.

Connect with Eric Johnson

![](https://cdn.prod.website-files.com/68c9c3107effc2ea46e1a82c/68c9c3107effc2ea46e1b26a_Mike.jpg)

Mike Edwards

Mike has spent over 25 years working in the fields of IT, corporate operations, and software development - most recently at Apple. Mike volunteers his time serving as a Board member with the [LLVM Foundation](https://foundation.llvm.org/), focusing on finance and operations. Mike truly believes in the power of AI to help address some of the world’s greatest needs.

Connect with Mike Edwards

Return to page

No items found.

![](https://px.ads.linkedin.com/collect/?pid=3780916&fmt=gif)

 ![](https://px.ads.linkedin.com/collect/?pid=8130124&fmt=gif)