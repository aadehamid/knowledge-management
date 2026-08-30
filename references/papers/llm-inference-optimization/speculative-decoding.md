title: How speculative decoding makes LLMs go brrr – Leonie Monigatti
description: Learn how speculative decoding accelerates LLM inference without quality loss: the draft-and-verify algorithm, plus Medusa, EAGLE, DFlash, and DSpark.

# How speculative decoding makes LLMs go brrr – Leonie Monigatti

Transformer-based Large Language Models (LLMs) generate text autoregressively. This means that text is generated token by token, where each token requires a full forward pass of the model. This sequential dependency makes inference latency proportional to output length, and thus slow. For latency-critical applications, such as real-time conversation or multi-turn agentic workflows, this is a real bottleneck in production speculative decoding aims to overcome.

**Speculative decoding** is an inference optimization technique that reduces decoding latency while preserving output quality. It was introduced by Chen et al. (2023, DeepMind) \[1\] and Leviathan et al. (2023, Google) \[2\], who proposed a similar approach independently around the same time. ![LLM Inference comparison with and without speculative decoding.](https://leoniemonigatti.com/blog/images/speculative-decoding/speculative-decoding-comparison.gif)

## How does speculative decoding work?

Speculative decoding speeds up generation by using a **draft-then-verify** approach. This approach contains two main components:

- **Target model**: the large, high-quality model whose output we want to accelerate.
- **Draft mechanism**: a smaller, faster model or mechanism that can approximate the target model’s distribution.

The [draft mechanism](#draft-mechanism) proposes a set of candidate tokens, which are then [verified in parallel](#verification-mechanism) by the target model in a single forward pass. Because the acceptance rule preserves the target distribution using [rejection sampling](#rejection-sampling), speculative decoding speeds up generation without any quality loss.

![](https://leoniemonigatti.com/blog/images/speculative-decoding/speculative-decoding.png "Speculative decoding overview.")

### Draft mechanism

In the first step, next token-prediction is offloaded to a less resource-intensive draft model or mechanism, which generates a short sequence of \\(\gamma\\) candidate tokens \\(x\_1, ... x\_{\gamma}\\) (typically 3 to 12). This leaves the expensive target model to verify tokens rather than generate each one.

The intuition behind this idea is that not every token is equally difficult to predict. Many next tokens are easy to predict (e.g., completing a common phrase or closing a bracket), and a small model can guess these as well as a target model. In contrast, tokens for specific facts or rare words are harder to predict and require a more capable model.

Speculative decoding exploits this imbalance by cheaply handling the easy tokens with the lightweight draft mechanism and catching the difficult ones that need the target model’s full judgment during verification.

### Verification mechanism

Once the draft tokens are proposed, the target model verifies them all simultaneously in a single forward pass.

This **parallel verification** is the key efficiency improvement:

Running the target model over \\(\gamma\\) drafted tokens at once costs roughly the same as running it over a single token, because the weights only need to be loaded once for the whole batch. Instead of paying the full memory-bound cost \\(\gamma\\) times, speculative decoding pays it once and verifies \\(\gamma\\) tokens for that price, exploiting GPU parallelism, which is underused in standard decoding.

### Rejection sampling

Finally, the target model verifies the draft tokens using **rejection sampling**. This process accepts the longest prefix consistent with the target model’s own distribution.

For each draft token \\(x\_i\\), we compare the target model’s probability \\(p(x\_i)\\) against the draft model’s probability \\(q(x\_i)\\). The draft token is accepted with probability \\(\alpha\_i\\). \\\[ \alpha\_i \= \min\left(1, \frac{p(x\_i)}{q(x\_i)}\right) \\\]

That means, if the draft model underestimates a token that the target likes, it’s accepted. If the draft model overestimates a token, it may be rejected. ![Verification with rejection sampling. Once a token is rejected, every token after it is discarded regardless of its own acceptance probability.](https://leoniemonigatti.com/blog/images/speculative-decoding/parallel-verification.png)

Depending on whether a token is accepted or rejected, the cycle ends differently:

- *If all draft tokens are accepted,* the target model appends a bonus token, generated during the verification pass.
- *If a draft token is rejected,* it and all its following draft tokens (even if they would have been correct) are discarded. The target model corrects the rejected token by resampling from the residual distribution: \\\[ p'(x) \= \text{norm}(\max(0, p(x) - q(x))) \\\]

The **acceptance rule** and **resampling correction** ensure that a draft token is accepted only if the target model would have generated it too, so that there’s no loss in accuracy.

To see why this holds exactly, split the target’s probability mass into two parts: the overlap it shares with the draft distribution \\(q\\), and the leftover mass where the target wants a token more than the draft did (\\(p > q\\)). Accepted draft tokens cover the overlap, since those are the tokens both models agree on. The leftover mass is covered by the resampling step, which draws corrected tokens whenever a draft is rejected. Together, accepted and resampled tokens reconstruct the target distribution \\(p\\) exactly, which is why speculative decoding introduces no quality loss.

![](https://leoniemonigatti.com/blog/images/speculative-decoding/rejection_sampling.png "Rejection sampling preserves the target distribution \\(p\\): accepted drafts cover the overlap (grey), rejections are corrected from the residual mass where \\(p > q\\) (diagonal), and the draft’s excess where \\(q > p\\) is discarded (crosses).")

## Why speculative decoding is fast

Autoregressive generation is memory-bound, not compute-bound. Each pass reloads weights and synchronizes memory, making low-batch inference memory-bandwidth bound and underutilizing the GPU’s parallel compute capacity.

Speculative decoding is fast because it turns that idle compute into additional tokens. The target model verifies all \\(\gamma \+ 1\\) positions in one forward pass, so a single expensive pass produces several tokens instead of one.

Two quantities are useful in practice:

- **Acceptance rate** \\(\alpha\\): the average probability that a draft token is accepted, equivalently the fraction of proposed tokens that get accepted. This is the per-token acceptance probability \\(\alpha\_i\\) from [rejection sampling](#rejection-sampling), averaged over positions and contexts.
- **Accepted length** \\(\tau\\): the expected number of tokens emitted in one speculative cycle, including the target token emitted after a rejection or after a fully accepted draft.

For a speculation length \\(\gamma\\) and average acceptance rate \\(\alpha\\), and under the simplifying assumption that acceptance is independent across positions, Leviathan et al. give the expected accepted length \\(\tau\\) as:

\\\[ \tau \= \sum\_{i\=0}\^{\gamma} \alpha\^i \= \frac{1 - \alpha\^{\gamma \+ 1}}{1 - \alpha} \\\]

The \\(i \= 0\\) term accounts for the token the target model always emits during verification (a correction after a rejection, or a bonus token after a fully accepted draft): even if the first draft token is rejected, the cycle still generates one token. If all \\(\gamma\\) draft tokens are accepted, the verification pass generates one additional target token. The accepted length \\(\tau\\) therefore ranges between 1 and \\(\gamma \+ 1\\).

Although the speedup depends on many other factors, such as batching and memory overhead, the following formula, as presented by Sadhukhan et al. \[3\], is a useful approximation of the average per-token latency \\(L\\), in terms of the time spent drafting (\\(T\_\text{draft}\\)), the time spent verifying (\\(T\_\text{verify}\\)), and the accepted length \\(\tau\\):

\\\[ L \= \frac{T\_\text{draft} \+ T\_\text{verify}}{\tau} \\\]

This approximation illustrates how speculative decoding improves decoding speed when:

- drafting becomes more accurate (\\(\tau\\) increases),
- drafting becomes faster (\\(T\_\text{draft}\\) decreases),
- or verification becomes faster (\\(T\_\text{verify}\\) decreases).

## Drafting strategies for speculative decoding

The latency approximation formula shows that speculative decoding is mostly a drafter-design problem: On the one hand, a smart but slow drafter reduces the latency speedup speculative decoding promises. On the other hand, a drafter that’s fast but often wrong wastes computation on drafting tokens the target model rejects.

Much of the modern literature focuses on navigating this trade-off: *how to make drafting both fast and accurate*.

### Independent draft model

The original implementations in Chen et al. \[1\] and Leviathan et al. \[2\] use a two-model system with a smaller draft model and a target model. The draft model is typically a lower-parameter model from the same model family (same tokenizer, similar instruction-tuning) as the target model. Later work also distills the draft model from the target to raise acceptance rates.

![](https://leoniemonigatti.com/blog/images/speculative-decoding/Independent.png "Drafting mechanism of an independent draft model.")

Using an independent draft model can reportedly speed up generation by 2-3x by using off-the-shelf models without additional training. However, this approach requires hosting a second model. Additionally, because the draft model is also autoregressive, speedup can decline past an optimal draft length because each extra draft token adds another sequential draft step while contributing diminishing acceptance gains.

### Medusa

Cai et al. \[4\] introduced Medusa, which eliminates the operational cost of hosting a separate draft model entirely. Instead, it builds the drafting mechanism directly into the target model by augmenting it with **multiple lightweight prediction heads**, where the \\(i\\)-th head predicts the token \\(i\+1\\) positions ahead (the target model’s own LM head already covers the next token). The resulting candidate continuations are organized into a tree and verified in parallel using **tree attention**.

![](https://leoniemonigatti.com/blog/images/speculative-decoding/medusa.png "Drafting mechanism of Medusa.")

The Medusa-1 approach reports around 2.2x speedup, while the Medusa-2 approach reports 2.3-3.6x speedup and an acceptance length of 3.0-3.5 tokens per step. Although this removes the cost of hosting a second model, the extra heads require additional training. Additionally, because each head predicts its position independently of the others, draft quality decays at later positions, lowering acceptance rates.

### Multi-token prediction (MTP)

Multi-token prediction (MTP), introduced by Gloeckle et al. \[11\], was not designed as a speculative decoding technique, but to *train a better model*. We include it for completeness. MTP uses the same multi-head idea as Medusa, adding extra output heads that each predict a token further ahead. Predicting several tokens at once is a richer learning signal that makes the model itself stronger. The inference speedup of up to 3x is just a side effect.

The different purpose drives the main differences from Medusa:

- **Goal:** Medusa bolts heads onto a finished model to draft faster, while MTP trains the heads during pretraining to make the base model better, and gets a drafter for free.
- **Base model:** Medusa leaves the model’s quality unchanged, while MTP improves it.
- **Applicability:** Medusa can be added to any off-the-shelf checkpoint, while MTP has to be included from the start.

### EAGLE / EAGLE-2 / EAGLE-3

The EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) series was introduced by Li et al. \[5\] and extended in EAGLE-2 \[6\] and EAGLE-3 \[7\]. It’s a speculative decoding method that drafts at the feature level, extrapolating from the target model’s internal hidden states.

Although folding the drafter into the target removes the overhead of maintaining a second model, Medusa’s token-level heads are independent, which means the drafter re-derives context the target already computed head by head. EAGLE pushes the built-in-drafter idea one step further: instead of predicting future tokens directly, it exploits and reuses the feature-level context from the frozen target model:

- **EAGLE** (2024) \[5\] predicts the next feature (hidden state), rather than the next token directly, to boost acceptance rates. For this, it attaches a small autoregressive drafter to the target model that operates at the **feature level**: It takes the target model’s top-layer features (immediately before the LM head) together with the embedding of the previously sampled token. It fuses the two to predict the feature of the next position. The model then passes this predicted feature through its frozen LM head to generate a draft token. This reportedly speeds up inference by 2.7-3.5x (reported for LLaMA2-Chat 70B).
- **EAGLE-2** (2024) \[6\] introduces **dynamic draft trees** which adapt the tree structure to the drafter’s confidence. This lets the drafter explore multiple generation paths, producing longer branches of predictable text and shorter ones for complex parts. This reportedly speeds up inference by 3.05-4.26x.
- **EAGLE-3** (2025) \[7\] refines the training objectives by replacing feature prediction with direct token prediction, enabled by a *training-time test* that trains the drafter on its own fed-back outputs to close the train/inference input mismatch. For this, EAGLE-3 fuses features from multiple target layers instead of relying on the top layer alone. This reportedly scales speedups up to 6.5x.

![](https://leoniemonigatti.com/blog/images/speculative-decoding/EAGLE-3.png "Drafting mechanism of EAGLE-3.")

However, EAGLE’s draft mechanism is still autoregressive, meaning drafting cost grows with draft length, and errors accumulate across the block. The following drafters with parallel approaches aim to overcome this.

### DFlash

DFlash, introduced by Chen et al. \[8\], is a speculative decoding framework that uses a lightweight block diffusion model for parallel drafting. Like EAGLE, DFlash treats the target model’s internal features as a rich drafting signal, but instead of drafting one token at a time, it generates a whole block in parallel. DFlash combines the speed of a parallel block-diffusion model for drafting with the quality of an autoregressive target model for verification.

Nearly all drafting variants above are **autoregressive**. Yet the sequential nature of autoregressive drafters creates the same bottleneck as autoregressive decoding. Under DFlash’s own benchmarks, this caps autoregressive drafters at roughly 2-3x.

An effective alternative to autoregressive generation is to use diffusion LLMs for **parallel** generation. The trade-off, however, is that current diffusion models typically generate lower-quality outputs than autoregressive models. Especially, fully parallel diffusion models suffer from fixed-length generation and lack efficient KV cache support. **Block diffusion models** \[9\] address these issues by denoising blocks of masked tokens simultaneously. This enables parallel generation while narrowing the quality gap to autoregressive models.

To achieve this, the goal is to train the draft models to align with the target distribution. This is possible because the target model’s hidden features encode information about future tokens and capture long-range dependencies. Conditioning the drafter on these context features lets it predict future blocks of tokens with high acceptance rates. Concretely, DFlash does this in the following steps:

1. **Extracting context features:** To extract the context features, the target model first performs a standard prefill pass for a given prompt to generate the first token (anchor token). During this pass, the hidden representations from a fixed set of layers are extracted. These extracted hidden states are then concatenated and passed through a projection layer to fuse them into a compact target context feature.
2. **Target-feature conditioning via KV injection:** The fused context feature is injected into the draft model’s KV cache and used for each layer’s attention mechanism. This lets accepted length keep improving as more draft layers are added rather than plateauing.
3. **Block-parallel diffusion drafting:** The diffusion drafter generates an entire block of future tokens in a single forward pass by denoising all masked positions in parallel. Each draft block begins with an anchor token (a known token the drafter conditions on to predict the rest of the block). During training, DFlash samples these anchors from the ground-truth response and masks the `block_size − 1` positions that follow each one. The draft model then predicts those masked tokens in parallel.

![](https://leoniemonigatti.com/blog/images/speculative-decoding/DFlash.png "Drafting mechanism of DFlash.")

DFlash reports up to over 6x speedup on its best benchmarks, with average speedups closer to 4-5x. Its accepted length also scales effectively with the number of draft layers, meaning deeper drafters improve acceptance rather than plateauing.

Because a parallel drafter produces every draft position in a single forward pass, its drafting latency is nearly independent of the block size. This reduces drafting latency and achieves much higher hardware utilization, even with deeper draft models. In principle, this lets parallel drafters generate longer draft blocks, and higher-quality drafts in turn raise acceptance rates. However, in practice they often suffer from rapid acceptance decay because they lack inter-token dependencies.

### DSpark

DSpark was introduced by Cheng et al. \[10\]. It shows that speculative decoding isn’t only a drafting problem but also a verification problem at scale. Thus, the DSpark speculative decoding framework combines fast parallel generation with adaptive, load-aware verification by using a semi-autoregressive architecture.

Parallel drafters have two main downsides:

- **Generation quality:** Because parallel drafters predict each position in the block independently, they lack inter-token dependencies. Although early draft tokens may be strong, this causes later positions to suffer from suffix decay.
- **Verification waste:** Although parallel generation can quickly produce long draft blocks, they’re only useful if the draft tokens will be accepted. Under high concurrency, verifying long draft blocks with a high rejection risk wastes critical batch capacity of the target model that could serve other requests.

DSpark keeps the parallel advantage but adds just enough sequential structure with two core ideas:

1. **Semi-autoregressive generation:** It combines a **parallel backbone** with a **lightweight sequential head** to model intra-block dependencies and mitigate suffix decay. The parallel backbone generates hidden states and base logits for a draft block, while the sequential head samples left to right inside the block and adds a prefix-dependent transition bias. This allows the drafter to get most of the block-parallel speed still, but each sampled token can depend on previous draft tokens. ![Drafting mechanism of DSpark. (Verification scheduler not depicted)](https://leoniemonigatti.com/blog/images/speculative-decoding/DSpark.png)
2. **Confidence-scheduled verification:** DSpark also decides how much of each draft to verify per request, instead of using a fixed length. A **confidence head** estimates how likely each draft token is to survive verification, and a **hardware-aware prefix scheduler** combines those estimates with the engine’s current load and throughput to choose how many draft tokens to verify for each request.

The DSpark approach increases accepted lengths, reduces verification waste, and increases per-user generation speeds by 60-85% at matched throughput.

However, DSpark’s gains come at the cost of complexity: it adds a sequential head, a confidence head, and a load-aware scheduler, along with the calibration and serving integration they require. Its scheduling is also only as reliable as its confidence estimates, which can drift under distribution shift.

## Speculative decoding in practice

Whether speculative decoding pays off in practice depends on the conditions the target model runs under.

At **low batch sizes**, autoregressive decoding is memory-bandwidth bound. Each step reloads the model’s weights from memory while the GPU’s arithmetic units sit mostly idle (see [Why speculative decoding is fast](#why-speculative-decoding-is-fast)). Speculative decoding turns that idle compute into extra tokens, verifying \\(\gamma \+ 1\\) positions for roughly the cost of one. This is where it delivers its largest wins.

As **batch size grows**, whether that still holds depends on context length. With many short-context requests in flight, the target model’s matrix multiplications become compute-bound rather than memory-bound, so verifying the extra draft tokens is no longer close to free: it competes for the same saturated compute already serving other requests, the speedup shrinks, and verifying draft tokens that are likely to be rejected can even reduce overall throughput.

When it does pay off, speculative decoding is available out of the box in most major serving stacks, such as [vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/), [SGLang](https://docs.sglang.io/docs/advanced_features/speculative_decoding), [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp), or [MLX](https://github.com/lmstudio-ai/mlx-engine/blob/main/mlx_engine/model_kit.py). Each lets you attach a draft model, and several also support EAGLE- or Medusa-style drafters, behind a few configuration flags, for example:

```
python -m sglang.launch_server \
  --model-path <target-model> \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path <draft-model>
```

## Summary

Speculative decoding accelerates LLM inference without changing the output by combining a lightweight drafting mechanism with parallel verification through the target model to preserve the target distribution. For this to work effectively, the drafter must be both fast and accurate.

Recent speculative decoding approaches study this trade-off from a separate draft model, to drafting heads built into the target (Medusa, MTP), to feature-level autoregressive drafters that reuse the target’s own hidden states (EAGLE series), to parallel block-diffusion drafters that generate a whole block at once (DFlash), and finally to semi-autoregressive drafting paired with load-aware verification scheduling (DSpark).

Most recently, DSpark also marks a shift in framing: past a certain serving scale, speculative decoding is no longer only a drafting problem but also a verification-scheduling one.

| Independent draft model | Autoregressive | \~3.6 tokens | 2-3x |
|----|----|----|----|
| Medusa | Parallel (multi-head) | \~3.0-3.5 tokens (Medusa-2) | 2.2-3.6x |
| EAGLE-3 | Autoregressive | \~5-7.5 tokens | 6.5x |
| DFlash | Block-parallel diffusion | \~4-8 tokens | >6x |
| DSpark | Semi-autoregressive | \~3.1-6.2 tokens | 1.6-1.85x\*\* |

\* Acceptance lengths are usually reported on different models and datasets, so they’re indicative rather than directly comparable.

\*\* DSpark doesn’t report a wall-clock speedup against standard (non-speculative) autoregressive decoding. This figure is per-user generation speedup against the MTP-1 production baseline.

## References

\[1\] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper (2023). [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318).

\[2\] Yaniv Leviathan, Matan Kalman, Yossi Matias (2023). [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html). ICML 2023.

\[3\] Rohan Sadhukhan, Jian Chen, Zheyu Chen, Vikram Tiwari, Ruihang Lai, Jiayu Shi, I. En-Hsu Yen, Avner May, Tianqi Chen, Beidi Chen (2025). [MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding](https://proceedings.iclr.cc/paper_files/paper/2025/hash/13f972adf12bdf886583d48cd528002f-Abstract-Conference.html). ICLR 2025.

\[4\] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao (2024). [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://proceedings.mlr.press/v235/cai24b.html). ICML 2024.

\[5\] Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang (2024). [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://proceedings.mlr.press/v235/li24bt.html). ICML 2024.

\[6\] Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang (2024). [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://aclanthology.org/2024.emnlp-main.422/). EMNLP 2024.

\[7\] Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang (2025). [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://openreview.net/forum?id=4exx1hUffq). NeurIPS 2025.

\[8\] Jian Chen, Yesheng Liang, Zhijian Liu (2026). [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036).

\[9\] Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, Volodymyr Kuleshov (2025). [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html). ICLR 2025.

\[10\] Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong, et al. (2026). [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147).

\[11\] Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve (2024). [Better & Faster Large Language Models via Multi-token Prediction](https://proceedings.mlr.press/v235/gloeckle24a.html). ICML 2024.
