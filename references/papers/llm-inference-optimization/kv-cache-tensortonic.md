title: KV Cache in LLMs Explained: How Transformers Make Inference Fast
description: Learn how the KV cache makes LLM inference fast. Prefill vs decode, memory math, GQA, MQA, PagedAttention, and the geometry behind autoregressive transformers.
author: TensorTonic Team
keywords: kv cache, kv cache llm, kv cache explained, kv cache transformer, kv cache tutorial, kv cache attention, kv cache memory, kv cache size, kv cache formula, kv cache visualization, how kv cache works, llm kv cache, transformer kv cache, llm inference optimization, llm inference speed, llm memory bandwidth bound, llm serving, autoregressive decoding, autoregressive generation, causal attention, self attention cache, attention key value cache, attention caching, memoization llm, prefill vs decode, prefill decode llm, time to first token, ttft, tpot, time per output token, grouped query attention, gqa, mqa, multi query attention, multi head attention, pagedattention, vllm, llama inference, gpt inference, gpt-2 kv cache, llama 2 kv cache, llama 3 inference, transformer inference, decoder only transformer, how llms generate text, llm batching, kv quantization, llm interview

# KV Cache in LLMs Explained: How Transformers Make Inference Fast

## Introduction

During autoregressive generation a transformer recomputes attention against the entire sequence at every step, and most of that work is redundant: the keys and values for tokens already processed are deterministic and do not change during the run. The KV cache is the optimization that exploits this redundancy. The rest of this chapter is about what is being cached, why caching it is mathematically safe, how much memory it costs, and the techniques that have grown up around it to keep that cost in check.

As you generate more text, GPU memory usage grows. For a single short request the model weights still dominate, but as context length and batch size grow, the KV cache can exceed the weight footprint and becomes the limiting factor in production serving workloads.

Long-context requests are also more expensive to serve in practice, which shows up indirectly in inference provider pricing. The KV cache is one of several reasons (memory bandwidth, scheduling, batching headroom all come into play), but it is the one that can comfortably scale context length where every prompt token has its keys and values cached, holding GPU memory for the entire duration of the request.

#### What You Will Learn

- Why autoregressive decoding creates redundant computation
- Exactly which vectors get cached and why they are safe to cache
- How the cache is structured inside a real transformer (per-layer, per-head)
- How to calculate KV cache memory for any model and context length

## How LLMs Generate Text

Before we can understand what the KV cache solves, we need to understand how language models generate text. The process is called **autoregressive decoding**, and it works one token at a time.

#### The Generation Loop

1. 1 Take the full sequence of tokens so far (the prompt plus any tokens already generated)
2. 2 Run the entire sequence through the transformer forward pass
3. 3 Take the output at the last position, apply softmax over the vocabulary, sample the next token

Formally, the model computes a conditional probability distribution at each step:

$P(x_{t}\midx_{1},x_{2},\ldots,x_{t-1})$

The probability of each possible next token, given every token that came before it.

The conditional is on *every token from 1 to t-1*, not just the last one. A transformer does not predict the next token in isolation; it attends to the full history at every step. That is what lets these models stay coherent across thousands of tokens, but it is also why every generation step has to chew through the entire sequence again.

#### The Fundamental Tension

The model requires all previous tokens to predict the next one correctly. This is what gives transformers long-range coherence. As the sequence grows, each generation step processes more tokens, and that cost compounds fast.

## Interactive: Generation Loop

Click **Next Token** to step through autoregressive generation. Every token in the context gets reprocessed on each step, and the attention matrix gains one row and one column per token.

KV cache: the recomputation problem total work \= O(T²) without cache prefill complete

## The Quadratic Problem

Without any optimization, each generation step runs the entire sequence through the transformer. At step 1 we process 1 token. At step 2 we process 2. At step t we process t tokens. The problem compounds at the attention layer.

Inside self-attention, every token attends to every other token. That means computing an n × n attention matrix at each step. But we already computed the attention scores for tokens 1 through t-1 at the previous step. We throw all of that away and start from scratch.

#### Concrete Trace of Wasted Work

Step 1: Compute K₁, V₁. Run attention over 1 token.

Step 2: Compute K₁, V₁, K₂, V₂. K₁ recomputed!

Step 3: Compute K₁..K₃, V₁..V₃. K₁, K₂ recomputed again!

Step t: Compute K₁..Kₜ, V₁..Vₜ. Tokens 1 through t-1 all recomputed.

Total work across T steps: 1² \+ 2² \+ ... \+ T² \= O(T³)

For a 1,000-token response, this is roughly 333 million attention score computations. For 4,096 tokens, about 22 billion. The cubic scaling makes long contexts progressively more punishing.

#### Why This Matters in Practice

In practice the KV cache is the difference between an inference path that is usable and one that is not. Exact speedups depend on the model, the GPU, the batch size, and the implementation, but the qualitative picture is the same everywhere: as the sequence gets longer the gap between the cached and uncached paths widens fast, because each new step in the uncached path is redoing more work than the step before it.

## Interactive: The Quadratic Problem

Compare the work done with and without KV caching. The gap between the two curves widens fast as the sequence gets longer.

The Quadratic Problem

each cell \= recomputing 1 token's attentioneach cell \= reading 1 cached K,V pair

T 6

## Attention Review: Q, K, V

Each token embedding is projected into three vectors via learned weight matrices, then attention is computed as a scaled dot product:

Q \= X · W ~Q~ "What am I looking for?"K \= X · W ~K~ "What do I offer?"V \= X · W ~V~ "What do I contain?"

$\text{Attention}(Q,K,V)=\text{softmax}\text{⁣}(\frac{QK^{T}}{\sqrt{d_{k}}})V$

During decoding, Q, K, and V have different shapes. This asymmetry is why caching works:

Current token only. One vector per step.

All context tokens. Grows with t.

All context tokens. Mixed by attention weights.

Q, K, V Projections STEP 3

## The Key Observation: K and V Stay Valid

K and V are not immutable in any deep mathematical sense; they happen to stay valid throughout causal decoding because of two things at once: the weights of the model are frozen at inference time, and the causal mask prevents any future token from feeding back into a past one's hidden state. Together those two facts mean that once you have computed K\_i and V\_i at some layer, no later step in the decode will ever need a different value for them. Look at how K\_i and V\_i are formed at layer ℓ:

K\_i at layer ℓ depends only on the hidden state h\_i^(ℓ)^ entering that layer for token i, and the layer's frozen weight matrix W\_K^(ℓ)^.

Once we compute K\_i and V\_i for token i at a given layer, they will never change. The input to the projection at layer ℓ is the hidden state h\_i^(ℓ)^ that token i carries out of the previous layer, and under causal attention that hidden state is itself fixed the moment token i has been processed (no future token can flow back into it). The weight matrices W\_K and W\_V are fixed too, since training is over. With both factors frozen, K\_i \= h\_i^(ℓ)^ · W\_K is a deterministic value that will be identical every time it is computed.

Q is the only piece that changes at each step. Q\_t \= h\_t^(ℓ)^ · W\_Q is recomputed from the new token's hidden state. K and V for every previously processed token are bitwise identical to what we got when we first ran them through the model.

#### Which Quantity Changes at Each Step?

Q\_t Recomputed K\_1..K\_t-1 Cacheable V\_1..V\_t-1 Cacheable

#### One Important Clarification

This analysis applies to **decoder-only models** like GPT, where attention is causal: token i can only attend to tokens 1 through i. The causal mask is what makes the cache work, because it guarantees that no future token can reach back and change the hidden state of an earlier one. KV caching does not really come up for encoder-only models like BERT, but the reason is different than people sometimes assume: it is not that the K and V of older tokens would shift, it is that encoders are not run autoregressively in the first place. They take a fixed input, run all tokens through bidirectional attention in a single shot, and produce a representation for the whole sequence at once. There is no streaming loop for a cache to amortize over.

## Interactive: Q vs K/V Roles

Q is freshly computed at every step from the new token. K and V from previous tokens just sit in the cache, untouched. Step through to see the asymmetry.

Q, K, V Roles in KV Cache STEP 3

## How KV Cache Works

With the key observation established, the implementation is straightforward. Allocate a buffer to store K and V vectors as we compute them, and reuse them on every subsequent step instead of recomputing.

#### Pseudocode: Cached Decoding

```
# Initialize empty cache for each layer
K_cache = []   # grows to shape [t, d_k]
V_cache = []   # grows to shape [t, d_v]

for each new token x_t:
    # Only compute projections for the NEW token
    q_t = x_t @ W_Q     # shape [1, d_k]
    k_t = x_t @ W_K     # shape [1, d_k]
    v_t = x_t @ W_V     # shape [1, d_v]

    # Append to the cache
    K_cache.append(k_t)  # now shape [t, d_k]
    V_cache.append(v_t)  # now shape [t, d_v]

    # Full attention using cached context
    scores  = q_t @ K_cache.T / sqrt(d_k)  # [1, t]
    weights = softmax(scores)               # [1, t]
    output  = weights @ V_cache             # [1, d_v]
```

Each step now does work proportional to t (one Q against t cached K/V pairs), not t². We compute K\_t and V\_t once, store them, and never touch them again.

These complexities refer to attention operations only. The MLP and LayerNorm sublayers still cost O(d\_model) per token per step, which sums to O(T) per layer per decode step with the cache in place. The cache changes the attention term from cubic to quadratic; it does not make the rest of the transformer free.

#### The Core Insight

Each token corresponds to one row in K and one row in V. Once computed, those rows are permanent. KV caching is memoization of matrix-vector products: compute each one once, store the result, reuse it forever.

## Interactive: Decoding Animation

Step through the decode and see what stays and what gets recomputed. K and V land in the cache once per token; Q is computed fresh at every step.

KV Cache Decode Step

## Per-Layer, Per-Head: The Real Structure

So far we have talked about a single attention layer with a single head. Real transformers have many layers and many attention heads. The KV cache exists separately for every layer and every head.

#### Which Layers Interact Across Tokens?

Token Embedding Pointwise lookup. No cross-token interaction.

Layer Norm Normalizes each token independently.

Self-Attention Every token attends to every other. This is where the KV cache lives.

Feed-Forward (MLP) Applied independently to each token. No cross-token interaction.

Self-attention is the only operation where token i can see token j. Everything else is pointwise. That is why the KV cache only lives inside attention: it is the only place where past context needs to be retrieved. When using KV cache, the FFN and LayerNorm layers process only the new token, doing a constant amount of work per step.

In multi-head attention, each head has its own W\_Q, W\_K, and W\_V projection matrices. Each head produces its own K and V vectors, which must be cached separately. For a model with L layers and H heads per layer, the total cache is L × H separate K and V matrices:

Cache shape \= \[L layers\] × \[H heads\] × \[t tokens\] × \[d\_head\]

Stored for both K and V → multiply by 2

#### Example: GPT-2 Small

For a 1024-token context in FP16: 2 × 2 bytes × 12 × 12 × 64 × 1024 \= **36 MB**

Note: d\_model \= H × d\_head \= 12 × 64 \= 768. This is the standard relationship.

## Interactive: Layer Architecture

Here is where the cache actually lives in the transformer block. Embeddings, LayerNorm, and the feed-forward sublayer touch one token at a time. Attention is the only place that needs every previous K and V, which is why it is the only place the cache exists.

KV Cache in the Transformer STEP 4

## Prefill vs Decode

Every LLM inference request has two distinct phases. Understanding them is essential for reasoning about latency and system design.

- Processes all N prompt tokens in one parallel forward pass
- Builds the initial KV cache from scratch
- Compute-intensive: large matrix multiplications run in parallel
- Determines time to first token (TTFT)
- Complexity: O(N²) but fully parallelizable on GPU

- Generates one token per forward pass
- Appends one K, V vector to the cache each step
- Memory-bandwidth bound: reads the full cache each step
- Determines time per output token (TPOT)
- Complexity: O(t) per step, O(T²) total

During prefill, the model processes all prompt tokens simultaneously, just as it does during training. This is efficient because the GPU can fill its compute units with large matrix operations over the full sequence. During decoding, however, we generate one token at a time. Each decode step reads the entire KV cache but writes only a tiny amount. This pattern makes decoding **memory-bandwidth bound** rather than compute-bound.

#### Why the First Token Takes Longer

When you send a 4,096-token prompt, the model runs a full forward pass over all 4,096 tokens to generate the first output token and build the initial cache. That is the prefill step. Every subsequent token only requires one pass over a single input token, but it reads the entire 4,096-entry cache from GPU memory.

That streaming pattern in modern chat interfaces, where you wait briefly and then text pours out, is just prefill followed by decode. The pause is the model burning compute on the whole prompt at once. The fast streaming after is one cached forward pass per token.

#### Decode Is Memory-Bandwidth Bound

At each decode step, the GPU reads the entire KV cache from HBM (high-bandwidth memory) to compute attention. For a large model with a long context, this can be tens of gigabytes of data per step. The bottleneck is no longer floating-point operations but the speed at which data can be moved from memory to the compute units. This is why Grouped Query Attention and KV quantization matter so much for serving: they reduce the volume of memory read per step, directly improving throughput.

## Interactive: Latency Comparison

Watch the difference between prefill and decode phases.

Prefill and Decode Phases

## Memory Cost: The Math

The KV cache memory cost can be computed exactly. For standard multi-head attention:

KV Memory \= 2 × P  × L  × H  × d~head~  × S

Since H × d\_head \= d\_model, the formula simplifies in the standard case. But for models using Grouped Query Attention (GQA), you replace H with H\_kv, the number of KV heads, which is smaller:

2 × P × L × H\_kv × d\_head × S

#### Real Model Examples (FP16, 4096-token context, 1 sequence)

| Model | Layers | KV Heads | d\_head | KV Cache |
|----|----|----|----|----|
| GPT-2 Small (117M) | 12 | 12 | 64 | 144 MB |
| Llama-2 7B | 32 | 32 | 128 | 2 GB |
| Llama-2 70B (GQA) | 80 | 8 | 128 | 1.3 GB |
| GPT-3 (175B) | 96 | 96 | 128 | 18 GB |

Llama-2 70B (GQA, 8 KV heads vs 64 query heads) has a smaller KV cache than Llama-2 7B despite being 10x larger. GQA is extremely effective.

#### Worked Example: 30B Model, Batch of 128

180 GB just for KV cacheThe model weights are only 2 × 30B \= 60 GB in FP16

#### The Uncomfortable Reality

At batch size 128 and 1024-token context, the KV cache requires 3x more memory than the model itself. This ratio is typical for production inference. It is why inference providers charge more for long-context requests, why batch size limits throughput, and why the entire field of KV optimization (GQA, MQA, quantization, PagedAttention) exists.

## Interactive: Memory Calculator

Configure model parameters and see how KV cache memory scales.

KV Cache vs Model Weights SEQ LEN 512 BATCH 1 PREC.

## The Space-Time Tradeoff

The KV cache is a textbook space-time tradeoff. You spend memory to avoid spending compute. The complexity numbers make this concrete:

Cost paid: O(T × d\_model × L) memory for the cache

This is the same idea as dynamic programming. Instead of recomputing overlapping subproblems, store their results. Computing Fibonacci naively recomputes fib(2) an exponential number of times. Cache the results and each value is computed once. The KV cache applies exactly this principle to attention: each K\_i and V\_i is computed once and reused in all subsequent steps.

#### When Is It NOT Worth It?

- Training: Backpropagation requires all intermediate activations. You cannot throw them away and retrieve them from a cache.
- Encoders (BERT): Bidirectional attention means the K and V of token i change when new tokens are added, since all tokens attend to all others.
- Very short sequences: The overhead of managing the cache is not worth it for sequences of 10 to 20 tokens where the quadratic cost is negligible.

#### The OS Analogy

Modern operating systems cache recently-accessed disk pages in RAM to avoid slow disk reads. The KV cache does the same thing: it keeps recently-computed attention state in GPU memory to avoid slow recomputation. PagedAttention takes this analogy further and implements virtual memory paging for the KV cache, dividing it into fixed-size blocks that can be allocated non-contiguously.
