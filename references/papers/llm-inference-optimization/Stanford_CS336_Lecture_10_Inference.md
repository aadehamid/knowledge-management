# Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 10: Inference

**Source**: [YouTube](https://www.youtube.com/watch?v=EfM546A79aM)
**Course**: Stanford CS336 — Language Modeling from Scratch (Spring 2026)
**Instructor**: Percy Liang (Professor of Computer Science)
**Co-Instructor**: Tatsunori Hashimoto (Assistant Professor of Computer Science)
**Course Website**: https://cs336.stanford.edu/
**Playlist**: [Stanford CS336 Spring 2026](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## Lecture Overview

**Inference**: given a **fixed model**, generate responses given prompts.

### Topics Covered

1. Understanding the inference workload
2. Review of transformer math and arithmetic intensity
3. Arithmetic intensity of inference (prefill vs generation)
4. Throughput and latency analysis
5. Taking shortcuts (lossy): reducing KV cache, alternative architectures, quantization, model pruning
6. Lossless shortcuts: speculative sampling/decoding
7. Handling dynamic workloads: continuous batching, paged attention

## Transcript

So this is lecture 10.
We're going to take a brief respite from scaling laws.
And we're going to talk about inference.
So the question of inference is a very simple one.
Given a fixed model that we've trained, generate responses given prompts.
First, we're going to start by understanding what the implications of inference are.
And the workload that it entails.
And then we're going to talk about ways of making inference faster.
And throughout this lecture, you're going to see that there's a lot of-- inference is a very deep topic.
Actually, we didn't do inference last year in lecture.
So this is the first year we're doing it.
But there's actually many, many topics that could span multiple lectures which I'll try to condense into one.

So inference shows up in multiple different places.
The most obvious place is if you actually want to use a model, you want to use it to chat.
You're using cursor or something to do code completion.
If you're running batch data processing job using your language model, all of these cases demand inference because you need to generate tokens from your actual model.
But it also shows up in other contexts.
If you want to even evaluate your model, let's say on instruction following need to do inference.
There is a lot of interest in test-time compute, which means thinking more before you actually output the final answer.
And that's also more inference because thinking is basically generate tokens.
And then, finally, even training itself.
If you're using reinforcement learning, you need to sample responses and then evaluate them based on some reward and that also requires inference.
So inference isn't just I want to put up a chatbot demo.
Inference actually is going to underlie many of the basic functions of a language model.
And even though it's one lecture, I want to stress how important it is for many things.
And we'll probably come back to this when we talk about an alignment later in the class.

So now inference is important.
So the theme of this class is efficiency.
And efficiency clearly matters.
Training is a one-time cost, but inference you repeat multiple times.
So here's some anecdotal stats on why inference is a big deal.
So Sam says OpenAI generates 100 billion words a day, which is quite a lot.
And even Cursor, which is not that new of a product, is allegedly generating a billion lines of accepted code each day.
So it's just giving you an idea of how much inference is accounting for and it costs of inference compared to training are definitely increasing.

So how do you measure what good inference looks like? So there's Time-To-First-Token, TTFT.
So this is how long an individual user needs to wait before any generation happens at all.
And this matters clearly for interactive applications.
If you have a big prompt and then you have to wait there for 10 seconds, that may not be a good user experience.
Latency is how fast tokens are arriving after maybe the first token.
This also matters for interactive applications.
Throughput is something a bit different.
Throughput is how many tokens in general are generated per not for overall users.
So this is particularly useful in batch processing applications.
So you can think about the throughput is high throughput doesn't mean low latency, because some requests might just take a very long time and you still have high throughput.
Latency is like the worst case of an individual user.

So what do you need to think about when you think about the efficiency of inference?
So in training, the key idea is that you get to see all the tokens, at least a supervised training, which means that you can parallelize over the sequence.
This is exploited heavily in the transformer.
So you've done the transformer training.
You know that you basically construct these tensors over the entire sequence.
And it's just like tensor matmuls and then you get your output.
But the key defining feature of inference, at least for transformers, is that you have to generate sequentially.
You can't parallelize because the generation of a token depends on all of the past.
So this is going to be the key thing that's going to make inference a lot harder.
And in particular, it's going to be harder to utilize all the compute that's available.
And it's going to be memory-limited, as we'll see in detail later.

So a lot of people are doing inference.
Anyone who's actually has a product and platform quickly realizes that these costs in doing large models is going to go up.
So they spend a lot of time and engineering effort trying to reduce that time.
So both providers serving close models and providers serving open weight models pay a lot of attention to inference.
More so than I think the average academic, because we're not actually serving any models.
We're just training and getting a score and putting in the paper.
But people who are actually serving models pay a lot of attention to inference.
So there's also a bunch of open source packages which are interesting to look at as well.

### Review of Transformer Math and Arithmetic Intensity

So I want to understand the inference workload in detail.
So I'm going to review briefly this transformer math that you did in assignment one and we talked a little bit about it during the first week of class.
So this is from the scaling jax-ml book, which is something you guys should really take a look at.
I think it does an excellent job of outlining many of the key concepts here.
And they have this really nice diagram that shows essentially the computation graph taken in input and having it go through the attention and the MLP layers.

In particular, we're going to use this notation so just to review this quickly.
So B is the number of sequences in your batch.
L is the number of layers.
T is the sequence length.
You can think about it as the number of tokens you're going to generate or query using.
S is also the sequence length, but how many you're kind of conditioning on in your prompt.
V is the vocabulary.
D is the dimensionality of the model.
F is the MLP hidden dimension, which is usually four times D.
H is the attention head dimension.
N is the number of query heads.
So generally N times H equals D, and then in GQA, Group Query Attention, you have a different number key value heads as query heads.
Usually K is smaller than N.
And G is the number of groups.
So K times G equals N.
And this diagram shows that you take your X.
You feed through the QKV matrices and you do a bunch of things.
So remember that the FLOPs required for a feedforward pass is 6 times the number of tokens, which is B times T times the number of parameters.
Plus for the attention there's another order of T.
So T times T is T squared dependence.

So let's also review arithmetic intensity, which is going to help us characterize when something is compute-limited versus memory-limited.
So just to start with the basic matmul.
So let's take a matrix X which is B by D and a matrix W, D by F.
And just to give some color to this computation, B is the batch size, D is the hidden dimension, and F is the projection matrix in the MLP.

So now let's do count the number of FLOPs and memory read and writes for just doing X times W.
So we're going to start with initialize to 0.
And what one has to do for this is we're going to read X from HBM.
So that means it incurs a memory cost of 2 times B times D assuming everything is in bf16.
You also read W. So that's 2 times D times F.
Then you do the matmul and that incurs 2 times B times D times F FLOPs.
So remember, this is from the first lecture.
So hopefully this is review.
And then you have to write it back out which is you have to pay another transfer.

So the total number of FLOPs is just the matmul.
And the number of bytes transferred is essentially the size of all the matrices that are read and written.
And arithmetic intensity is basically the ratio.
So the ratio is this expression.
And in general, just to simplify things a bit, generally the batch size is much less than D and F. B may be 100 and a D and F might be thousands or tens of thousands.
So I'm using SymPy here just to keep myself from making silly mistakes.
So basically, I'm letting C go to infinity and D scales as C times B and F scales as C times B.
And that gets you a simplified equation of B.
So the arithmetic intensity is B for this particular matrix multiplication.
So the way to interpret this is how many FLOPs are done per byte that was transferred?

So now the second part is you look at the accelerator, which for H100 flops per second is 989 teraflops, memory bandwidth 3.3 bytes per second.
And you divide, and that gives you what is called the accelerator intensity.
And if you look at the computation intensity, which is B, If it's greater than accelerator intensity, that means your compute-limited.
That means you're able to use all the GPUs or TPUs.
And if you're less than that, then your memory-limited, which is bad.
And so your compute-limited in this matrix multiplication case, if B is greater than 295 for a H100.
And all this is a bit idealized.
The actual details.
This is giving you a first-order approximation.

### Inference Workload: Prefill vs Generation

Two stages of inference:
1. **Prefill**: given a prompt, encode into vectors (parallelizable like in training)
2. **Generation**: generate new response tokens (sequential)

- Prefill is compute-limited (good) — easy to make compute-limited by making B*T large enough
- Generation is memory-limited (bad) — generating one token at a time (T=1), B is number of concurrent requests, hard to make large enough

The key bottleneck in generation is the **KV cache**: for every sequence (B), token (S), layer (L), head (K), store an H-dimensional vector. This grows linearly with sequence length.

### MLP Layer Arithmetic Intensity

For MLP layers (matrix multiplications only):
- Read X (B x T x D) from HBM
- Read Wup (D x F), Wgate (D x F), Wdown (F x D) from HBM
- Compute U = X @ Wup, G = X @ Wgate, Y = GeLU(G)*U @ Wdown
- Total FLOPs: 6·B·T·D·F
- Total bytes transferred: 4·B·T·D + 4·B·T·F + 6·D·F
- Arithmetic intensity simplifies to B·T (when B·T << D, F)

For the two stages:
- Prefill: easy to make compute-limited (good) by making B·T large enough
- Generation: T = 1, B is number of concurrent requests — hard to make large enough!

### Taking Shortcuts (Lossy)

#### Reducing KV Cache Size

- **GQA/MQA**: K/V vector sharing across heads reduces cache size
- **MLA (Multi-head Latent Attention)**: K/V projected to low-dimensional space C for caching, dramatic reduction
- **Sliding Window Attention (Local Attention)**: Attention fixed to recent K tokens, KV cache size constant regardless of sequence length

#### Quantization

Reduce model memory footprint. Int8 quantization handles outliers (abnormally large weights) in FP16/FP32 individually to maintain accuracy.

#### Model Pruning

Summary: reduce inference complexity without hurting accuracy.

From scratch recipe:
1. Define faster model architecture
2. Train faster model

Distillation recipe:
1. Define faster model architecture
2. Initialize weights using original model (which has a different architecture)
3. Repair faster model (distillation)

### Lossless Shortcuts: Speculative Decoding

Exploits the fact that checking (prefill) is faster than generation. A cheap draft model (P) generates ahead, and the target model (Q) verifies in parallel. This guarantees exact samples from the target model.

### Handling Dynamic Workloads

Batching over sequences in live traffic is tricky because:
1. Requests arrive at different times (waiting for batch is bad for early requests)
2. Sequences have shared prefixes (e.g., system prompts, generating multiple samples)
3. Sequences have different lengths (padding is inefficient)

#### Continuous Batching

Process requests as they arrive rather than waiting to form complete batches.

#### Paged Attention

Used in vLLM and similar systems. KV cache divided into blocks like OS virtual memory, addressing memory fragmentation.

### Summary

- Inference is important (actual use, evaluation, reinforcement learning)
- Different characteristics compared to training (memory-limited, dynamic)
- Techniques: new architectures, quantization, pruning/distillation, speculative decoding
- Ideas from systems (speculative execution, paging)
- New architectures have huge potential for improvement

## References

- [Scaling Book: Transformers](https://jax-ml.github.io/scaling-book/transformers/)
- [Scaling Book: Inference](https://jax-ml.github.io/scaling-book/inference/)
- [vLLM (Berkeley)](https://www.youtube.com/watch?v=8BaEwoTk8XI)
- [TensorRT-LLM (NVIDIA)](https://nvidia.github.io/TensorRT-LLM/overview.html)
- [TGI (Hugging Face)](https://huggingface.co/docs/text-generation-inference/en/index)
- [Course Lectures Repository](https://github.com/stanford-cs336/spring2025-lectures)
