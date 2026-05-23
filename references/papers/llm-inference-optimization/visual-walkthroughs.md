![Vizuara AI Labs](vizlogo.png)
Vizuara Visual Walkthroughs

[The book](#book)
[Foundations](#foundations)
[Basics](#basics)
[KV cache](#kv-cache)
[Eviction](#kv-eviction)
[Quantization](#quantization)
[Modern inference](#modern)
[Attention ♥](#attention)
[Companion](#workshop)

One library for the visual notebooks

# Inference engineering, explained visually.

A curated shelf of interactive walkthroughs for attention, KV cache reduction and eviction,
FFT convolution, Mamba, DeepSeek sparse attention, quantization, transformers,
and GPU anatomy.

How to use this page

Pick a card, open the walkthrough, and follow the story visually. The grouping is by
learning sequence: first the machine and transformer basics, then KV-cache reduction,
then cache eviction, then modern inference ideas, then attention as the heart of speeding inference.

**18**visual pages

**7**learning tracks

The book

## Read the full reference

Every walkthrough on this page is a visual companion to a chapter of the Inference Engineering book. Open the book itself for the full text.

[book28 chapters

### Inference Engineering

The full book in one HTML file. From the Roofline and the KV cache through FlashAttention, PagedAttention, quantization, speculative decoding, parallelism, and three capstone systems.

Open the book↗](book.html)

Track 0

## Foundations

The mathematical ideas that make model behavior measurable before the systems work begins.

[LLMsloss

### The Anatomy of Perplexity

A first-principles visual guide to perplexity loss, token probabilities, cross-entropy, and what the score means for language models.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/perplexity-anatomy.html)

Track 1

## Basics of inference

Start with the two foundations: what a transformer is, and what kind of machine runs it.

[transformerfoundation

### DNA of a Transformer

A visual grounding in the components and data flow that make transformer models work.

Open walkthrough↗](https://vizuaraai.github.io/dna-of-a-transformer/)
[GPUhardware

### Anatomy of a GPU

A hardware-first visual experience for understanding the machine underneath modern inference workloads.

Open walkthrough↗](https://brrrviz.com/)

Track 2

## Reducing KV cache across tokens

This track is about making long-context inference cheaper: shrink the cache, reuse computation, and understand how local context grows.

[KV cachetoken reduction

### Reducing KV Cache Across Tokens

The full visual lecture on how cache size changes as tokens are pruned, compressed, windowed, or summarized.

Open walkthrough↗](https://vizuaraai.github.io/kv-cache-token-reduction-walkthrough/visual-walkthroughs/kv-cache-token-reduction-visual-walkthrough.html)
[FFTconvolution

### Why Convolution Is N log N

A musical journey through reverb, convolution, frequency coefficients, and why FFT makes the computation fast.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/)
[sliding windowreceptive field

### Receptive Field Growth in Sliding Window Attention

How local windows can still see farther through layer stacking and relay paths across transformer layers.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/receptive-field-sliding-window-walkthrough.html)
[Mambaselectivity

### Why Mamba Needs Selectivity

Why a state-space model needs input-dependent behavior to decide what to remember, suppress, or pass forward.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/mamba-why-selectivity-matters.html)
[Mambaarchitecture

### Inside the Mamba Block

A step-by-step visual tour of the Mamba block, from projections and convolutions to selective scan.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/mamba-architecture-step-by-step.html)

Track 3

## KV Cache eviction

A focused walkthrough of dynamic KV-cache eviction: protected regions, attention scores, block removal, rotation, and representative retention.

[KV cacheeviction

### KV Cache Eviction Visual Walkthrough

How OpenVINO-style cache eviction keeps start and recent tokens, scores middle blocks, and evicts low-value memory during generation.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/kv-cache-eviction-walkthrough.html)

Track 4

## Quantization

From how a single float is stored in silicon, through the modern post-training methods (GPTQ, AWQ, GGUF) and quantization-aware training, to rotation-aware compression of the KV cache.

[quantizationlecture L06

### Quantization — A Visual Walkthrough

The full lecture on quantization, end to end: from how a float is stored in silicon, through symmetric and asymmetric mapping, the two PTQ schemes, GPTQ, AWQ, GGUF, QAT, and the 1-bit Johnson-Lindenstrauss trick for KV cache compression.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/quantization-visual-walkthrough/)
[quantizationrotation

### TurboQuant Visual Guide

A visual guide to rotation-aware quantization ideas and how they improve practical model compression.

Open walkthrough↗](https://vizuaraai.github.io/turboquant-rotation-visual-guide/)

Track 5

## Modern inference concepts

DeepSeek-style efficiency, sparse attention, and modern cache designs live here.

[DeepSeeksparse attention

### DeepSeek Sparse Attention

A focused visual explanation of sparse attention patterns and why they matter for efficient inference.

Open walkthrough↗](https://vizuaraai.github.io/DeepSeek-Sparse/)
[DeepSeek V4KV cache

### DeepSeek V4 KV Cache Visual Guide

A walkthrough of the KV-cache ideas behind DeepSeek-style inference efficiency.

Open walkthrough↗](https://vizuaraai.github.io/deepseek-v4-kv-cache-walkthrough/)
[RadixAttentionsemantic cache

### RadixAttention & Semantic Caching

Two cache layers above plain prefix caching. SGLang turns the cache into a tree so any matching prefix is reused; semantic caching skips inference entirely when a query *means* something already answered.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/radix-and-semantic-caching.html)

Track 6

## Attention is the heart of speeding inference

Attention is where the cost often concentrates, so this guide becomes the map for making inference faster.

[FlashAttentionFA1 → FA2

### FlashAttention-1 to FlashAttention-2

How the same exact attention math gets faster by changing the GPU schedule, tiling work, and exposing more independent Q-block jobs.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/flashattention-1-2.html)
[FlashAttention-3Hopper

### FlashAttention-3 Visual Walkthrough

How FA3 uses Hopper features like WGMMA and TMA to overlap memory movement, matrix multiply, and softmax work inside the attention kernel.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/flashattention-3.html)
[attentionoptimization

### Summary of Attention Mechanism Optimizations

A comparative visual map of attention variants and inference optimization strategies.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/attention-variants-inference-comparison.html)

Track 6

## Spine — the whole arc on one page

Single-page narrative companions to the book. Each one threads many chapters into one continuous story.

[arcspine

### From the Roofline to vLLM

The full arc on one page. Naive inference, the KV cache and its memory cost, compressing across heads (MHA→MQA→GQA→MLA), across tokens (sliding window, linear, Mamba), and FlashAttention as the kernel underneath — leading into the vLLM engine.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/roofline-to-vllm.html)
[attentionretraining

### Which Attention Variants Need Retraining?

FlashAttention is a systems rearrangement. MQA, GQA, and MLA compress across heads. Sliding window, sparse, and linear attention compress across tokens. A clean sort plus a 160M-parameter, 10B-token reproduction recipe.

Open walkthrough↗](https://vizuaraai.github.io/convolution-musical-journey/attention-variants-retraining.html)

Vizuara AI Labs

This page is a single launchpad for the visual walkthrough collection.

[Back to top](#top)