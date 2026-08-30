title: maxsim-cpu: Maximising Maxsim Efficiency
description: Introducing maxsim-cpu, a much faster way to compute the late interaction's MaxSim operator on modern CPU hardware, optimised for both x86 and Mac ARM.
author: Mixedbread Team
publisher: mixedbread ai inc.
keywords: Search API, NLP API, Machine Learning API, Vector Database, Store, Semantic Search, RAG, Retrieval Augmented Generation

# maxsim-cpu: Maximising Maxsim Efficiency

***TLDR:*** \
 [maxsim-cpu](https://github.com/mixedbread-ai/maxsim-cpu) provides a highly-optimised implementation of the maxsim algorithm, greatly speeding up scoring for ColBERT-like models.

In retrieval world, a lot of inference pipelines run on CPU: oftentimes because of cost optimization: CPU machines are pretty cheap and infinitely scalable. In a lot of cases, it's not even an optimization: if you're running a local retrieval pipeline, your laptop might not even have a GPU! In most cases, thankfully, CPUs can do a more-than-acceptable job for the typical retrieval workflow, where the only computations required are a single inference pass on a query followed by similarity calculations.

However, the equation is a bit different when it comes to multi-vector retrieval, which powers late-interaction models such as [ColBERT](https://arxiv.org/abs/2004.12832) or [ColPali](https://arxiv.org/abs/2407.01449). These models use the **MaxSim** operator, which requires considerably more computations than their single-vector counterparts (read on to know why!).

While modern hardware thankfully makes this pretty fast, the additional computational cost adds up: running MaxSim on \~1000 documents with `PyTorch` on CPU can add between **50 and 100 milliseconds** to each query compared to running it on GPU: while this may seem rather small at first glance, **inefficiencies add up**, and spending so much time on distance calculation can make it unviable for many latency-sensitive environments.

But it doesn't have to be this way! Obviously, current CPUs will never be as fast as GPUs for this sort of computation, but **they don't have to be so slow**! By taking advantage of architecture-specific instructions and low-level scientific computing libraries like `libxsmm`, we built **maxsim-cpu**: a small Python package, written in Rust, which cuts down the aforementioned overhead to just **5 milliseconds**.

![Jupyter timing cells: MaxSim over the same inputs takes 50.3 ms per loop in NumPy, 23.9 ms with Numba and 3.36 ms with maxsim-cpu](https://www.mixedbread.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Foverhead.1bwj0572xg1rj.png&w=3840&q=75&dpl=dpl_83tSPpNWbmGJ9pyzzm4NZQvsh6wL "MaxSim CPU overhead comparison"){width=1158 height=820}

> [!NOTE]- Show the data behind this chart
> | Implementation | Time per call (mean ± std of 7 runs, 100 loops each) |
> |----|----|
> | NumPy (`maxsim_np`) | 50.3 ms ± 1.02 ms |
> | Numba (`maxsim_nb`) | 23.9 ms ± 943 µs |
> | **maxsim-cpu (`maxsim_cpu.maxsim_scores`)** | **3.36 ms ± 171 µs** |

Want to just jump in and try it out? Check out our [GitHub repo](https://github.com/mixedbread-ai/maxsim-cpu) or install the library directly:

```
[uv] pip install maxsim-cpu
```

Want to understand a bit more about why it's useful? Read on!

## [What even is MaxSim?](#what-even-is-maxsim) {#what-even-is-maxsim}

MaxSim, or for **Max**imum **Sim**ilarity is the core element of current late-interaction models [ColBERT](https://arxiv.org/abs/2004.12832), [ColPali](https://arxiv.org/abs/2407.01449).... Its mechanism is simple: rather than performing a single cosine similarity computation between a vector representing the whole query and vectors representing entire documents, it computes **token-level** similarities.

For each candidate document, MaxSim iterates through every token within the query, and compares its similarity to **every token within the document**, before keeping the **maximum value** for each query token (hence the `Max`) and summing them up to produce a document-level score.

### [Orders of Magnitudes Matter](#orders-of-magnitudes-matter) {#orders-of-magnitudes-matter}

This approach has many benefits: it allows for capturing semantic relationships that larger-grain methods would miss. But it is inefficient: for each query, it requires thousands of similarity calculation. For a simple example: given 1000 candidate documents, each containing 300 tokens, and a 32-token long query , a "traditional" single-vector query search query would perform 1000 cosine similarity calculation: one for each document against the query. Using MaxSim, we require `n_query_token * n_docs * n_token_per_doc` distances, or `32 * 1000 * 300` \= **9 600 000** (yes, that's 9.6 **millions**) distance calculations.

![How MaxSim scores a document: every query token vector is compared with every document token vector, the maximum similarity per query token is kept, and those maxima are summed into the document score](https://www.mixedbread.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fmaxsim_visualised.2q23o29m_a2w7.png&w=3840&q=75&dpl=dpl_83tSPpNWbmGJ9pyzzm4NZQvsh6wL){width=3017 height=1032}

How is it viable, then? Well, thankfully, cosine similarity calcualtions are very computationally cheap. In fact, with normalized vectors (which everyone uses), it's a simple matrix multiplication. As you might know if you've ever looked at the math behind deep learning, matrix multiplications are the main computational operation that power all models, and GPUs are **very, very good** at running them quickly. This, compounded by the fact that individual vectors generated by ColBERT models are pretty small, means that the computational cost is pretty cheap: in fact, it's less half the compute required to run a forward pass through **a single layer of BERT-base**, something that any GPU released in the last decade can do in a handful of milliseconds.

## [The Problem](#the-problem) {#the-problem}

**So, there's no problem then?** Well, not really. GPUs **love** matrix multiplications and parallel computation, that's what they're built for: thousands of really weak cores. Their trustworthy cousin, the humble CPU, is not quite as big a fan of this sort of workload.

This results in the situation we mentioned in the introduction: computing MaxSim on CPU can quite often be a significant source of latency in retrieval systems. While the FLOPS required to perform these computations is relatively low, from the CPU's point of view, they're the most evil FLOPS there are: **a lot** of very small parallel computations. What CPUs enjoy is the complete opposite, they love performing **big, more demanding computations** that don't require so many tiny steps.

But CPU machines are very, very cheap and widely available. So, many pipelines end up just having to take that latency hit, or figure out workarounds so there are fewer documents to score with MaxSim.

## [The (Partial) Solution: maxsim-cpu](#the-partial-solution-maxsim-cpu) {#the-partial-solution-maxsim-cpu}

> But surely, CPUs can't be that slow? My scoring machine has a whole 48 cores, then it should be able to run scoring faster than in 60ms!?

Thank you for your question, convenient rhetorical question-asker. Indeed, no, it doesn't have to be so slow!

### [It always comes down to optimisation tradeoffs](#it-always-comes-down-to-optimisation-tradeoffs) {#it-always-comes-down-to-optimisation-tradeoffs}

A big reason as to why overhead is so big is that MaxSim-style computations, that is, a lot of very small matrices with very low vector dimensions, is quite simply something fairly uncommon and that very few major libraries actively seek to optimise for.

There **is** a good ecosystem of libraries to speed up CPU computations: ONNX, recent improvements in both PyTorch (better Intel MKL handling...) and JAX (XLA being pretty good at optimising CPU compute nowadays), but they're all (rightfully) much more interested in speeding up the kind of computations you need to run models.

There is actually an entirely separate ecosystem of matrix multiplication libraries which use what are, for the purpose of this blog, essentially magic tricks to speed up maxsim-style operations: they're spearheaded by the very clearly named [libxsmm](https://github.com/libxsmm/libxsmm) (SMM standing for Small Matrix Multiplication).

In `maxsim-cpu`, we built on top of libxsmm, and added some additional optimisations to speed things up further, such as fused operations to avoid having to load all the individual distances into memory (which is comparatively very slow, [as this visualisation shows](https://x.com/BenjDicken/status/1847310000735330344)), a separate code path to speed things up considerably when handling variable length documents, further optimisations to leverage Apple Silicon-specific optimisations rather than libxsmm, etc...

### [Gotta go fast](#gotta-go-fast) {#gotta-go-fast}

The result is the `maxsim-cpu` package, written in Rust and exposed as a Python library. In our rapidly ran tests, we observe considerable speed-up over other python packages, while being extremely low-dependency (all you need is numpy and maxsim-cpu itself):

![MaxSim speedup versus PyTorch when reranking 1000 documents on 32 cores of an AMD EPYC 9354, by document length](https://www.mixedbread.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fcpu.30xewqw534jnz.png&w=3840&q=75&dpl=dpl_83tSPpNWbmGJ9pyzzm4NZQvsh6wL){width=4169 height=2372}

> [!NOTE]- Show the data behind this chart
> Library512 tokens1024 tokens2048 tokensVariable (128–1536 tokens)PyTorch (baseline)1.0×1.0×1.0×1.0×PyTorch (matmul)1.0×1.0×1.1×1.0×NumPy1.4×1.4×1.25×1.05×JAX1.5×1.4×1.65×1.3×JAX (vmap)2.3×2.45×2.0×1.75×**maxsim-cpu (Rust)****10.0×****9.4×****9.1×****10.0×**Speedup versus PyTorch on CPU, reranking 1000 documents, 32 cores of an AMD EPYC 9354.

Even on MAC, with much fewer cores and no `libxsmm` to build on top of, we've observed noticeably speedups, especially on variable batch-size, as our approach allows us to do-away with the dreadfully slow and wasteful Python-based padding operations (note that you could probably optimise those for other libs to also be faster, but it's an annoying bit of engineering to design custom routing logic):

![MaxSim speedup versus PyTorch when reranking 1000 documents on an Apple M4 Max, by document length](https://www.mixedbread.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fmac.0067c9zz260cn.png&w=3840&q=75&dpl=dpl_83tSPpNWbmGJ9pyzzm4NZQvsh6wL){width=4170 height=2372}

> [!NOTE]- Show the data behind this chart
> Library512 tokens1024 tokens2048 tokensVariable (128–1536 tokens)PyTorch (baseline)1.0×1.0×1.0×1.0×PyTorch (matmul)1.0×1.0×1.0×0.85×NumPy0.65×0.75×0.75×0.85×JAX1.4×1.65×1.35×1.1×JAX (vmap)1.4×1.6×1.3×1.0×**maxsim-cpu (Rust)****2.0×****2.8×****2.9×****5.0×**Speedup versus PyTorch on an Apple M4 Max, reranking 1000 documents.

### [Using maxsim-cpu](#using-maxsim-cpu) {#using-maxsim-cpu}

The moment you've all been waiting for (or the moment you scrolled to if you're already familiar with MaxSim): how to get your hands on `maxsim-cpu`? Well, it's quite simple. Below are simple instructions, and you may find more detailed ones as well as the full source code in the [GitHub Repository](https://github.com/mixedbread-ai/maxsim-cpu).

#### [Installation](#installation) {#installation}

On Linux Machines with x86 processors that support AVX2 instructions (a lot of fancy words to say "Linux machines with a CPU released in the last decade") and Macs with Apple Silicon, you can install it directly from PyPi:

```
[uv] pip install maxsim-cpu
```

We do not currently support any other hardware nor do we have plans to, but contributions in this direction (adding AVX512-specific code paths to go even faster or supporting Windows) are welcome and the PRs will be reviewed.

For more detailed installation instructions, including building from source if you'd like to modify the library, head to [github](https://github.com/mixedbread-ai/maxsim-cpu).

#### [Usage](#usage) {#usage}

The library exposes two methods: `maxsim_cpu.maxsim_scores` and `maxsim_cpu.maxsim_scores_variable`, which you should route to depending on the nature of your input: `maxsim_scores` expects documents to all be the same length while `maxsim_scores_variable` allows variable length inputs. In all cases, each method expects a single query and its set of candidate documents. Usage is as follows:

```
import numpy as np
import maxsim_cpu

# Prepare normalized embeddings
query = np.random.randn(32, 128).astype(np.float32)  # [num_query_tokens, dim]

# NOTE: maxsim-cpu expects normalized vectors.
query /= np.linalg.norm(query, axis=1, keepdims=True)

docs = np.random.randn(1000, 512, 128).astype(np.float32)  # [num_docs, doc_len, dim]
# Normalize document embeddings...
docs /= np.linalg.norm(docs, axis=2, keepdims=True)

# Compute MaxSim scores
scores = maxsim_cpu.maxsim_scores(query, docs)  # Returns [num_docs] scores
```

Swapping in `maxsim_scores_variable` is straightforward:

```
import numpy as np
import maxsim_cpu

# Prepare normalized embeddings
query = np.random.randn(32, 128).astype(np.float32)  # [num_query_tokens, dim]

# NOTE: maxsim-cpu expects normalized vectors.
query /= np.linalg.norm(query, axis=1, keepdims=True)

# Create variable-length documents as a list
docs = [
    np.random.randn(np.random.randint(50, 800), 128).astype(np.float32)  # Variable length docs
    for _ in range(1000)
]
# Normalize each document in the list
docs = [doc / np.linalg.norm(doc, axis=1, keepdims=True) for doc in docs]

# Compute MaxSim scores
scores = maxsim_cpu.maxsim_scores_variable(query, docs)  # Returns [num_docs] scores
```

And that's pretty much it, that's all you need to know to use `maxsim-cpu`!

## [Conclusion](#conclusion) {#conclusion}

This blog post briefly introduces the MaxSim operator as well as our new package, `maxsim-cpu`. It is a standalone library, meant to do one thing and do it well, and is part of our efforts to open source any individual component we feel might be useful to more than just ourselves, as we previously did with [batched](https://www.mixedbread.com/blog/dynamic-batching) and [baguetter](https://www.mixedbread.com/blog/intro-baguetter). We hope it'll be useful to anyone who cares about MaxSim, and that it might even inspire more people to write more optimised versions of commonly used algorithms: search is more relevant than ever, but every small component can still be improved in so many ways.

If figuring out how to create these improvements sounds like something you're interested in, we are [currently hiring across all technical positions](https://mxbai.notion.site/job-board?pvs=74), don't be shy!

### [Citation](#citation) {#citation}

```
@online{maxsimcpu2025mxbai,
  title={{maxsim-cpu}: {M}aximising {M}axsim {E}fficiency},
  author={Benjamin Clavié and Sean Lee},
  year={2025},
  url={https://www.mixedbread.com/blog/maxsim-cpu},
}
```
