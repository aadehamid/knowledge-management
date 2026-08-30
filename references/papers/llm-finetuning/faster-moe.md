title: Fine-tune MoE Models 12x Faster with Unsloth | Unsloth Documentation
description: Train MoE LLMs locally using Unsloth Guide.

# Fine-tune MoE Models 12x Faster with Unsloth | Unsloth Documentation

We’re introducing \~12x faster Mixture of Experts (MoE) LLM training with **>35% less VRAM** and **\~6x longer context** with our new MoE Triton kernels and new mathematical optimizations, all with no loss in accuracy.

- Unsloth now supports fast training for MoE architectures including [gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune), [Qwen3](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune) (30B, 235B, VL, Coder), DeepSeek [R1](https://unsloth.ai/docs/models/tutorials/deepseek-r1-0528-how-to-run-locally), [V3](https://unsloth.ai/docs/models/tutorials/deepseek-v3.1-how-to-run-locally) and GLM ([4.6](https://unsloth.ai/docs/models/tutorials/glm-4.6-how-to-run-locally#glm-4.6v-flash), [4.7](https://unsloth.ai/docs/models/tutorials/glm-4.7), [Flash](https://unsloth.ai/docs/models/tutorials/glm-4.7-flash)).
- gpt-oss-20b fine-tunes in **12.8 GB VRAM**. Qwen3-30B-A3B (16-bit LoRA) uses 63GB.
- Our kernels work on both data-center (B200, H100), **consumer** and older GPUs (e.g., RTX 3090), and FFT, LoRA and QLoRA.

In collaboration with 🤗Hugging Face, we made all MoE training runs standardized with PyTorch’s new `torch._grouped_mm` function. Transformers v5 was recently optimized with \~6x faster MoE than v4 and Unsloth pushes this even further with custom Triton grouped‑GEMM \+ LoRA kernels for an **additional** \~2x speedup, >35% VRAM reduction and >6x longer context (12-30x overall speedup vs v4).

Try our Unsloth Notebooks for fast MoE training:

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FZYAbbmao7vQbKGr7Rtiv%252Fgraph%2520results%2520only.png%3Falt%3Dmedia%26token%3D6cfb4f27-3e78-48d3-b3db-12eb4b3dcde3&width=768&dpr=3&quality=100&sign=7962c66a91f308289908538479a9012a&sv=3){width=4000 height=1643}

### 🦥 Unsloth MoE Triton Kernels {#unsloth-moe-triton-kernels}

Alongside `torch._grouped_mm` (see [❓What is torch.\_grouped\_mm?](https://unsloth.ai/docs/basics/faster-moe#what-is-torch._grouped_mm)), we created custom Triton MoE kernels that can be even faster in some cases. They are also **backwards compatible** with older hardware like A100, and older PyTorch versions.

On A100, our **Triton kernels are \~2.5× faster** than `torch._grouped_mm`. The kernels also have a one‑time autotune step to pick the best kernel config.

Auto-tuning takes \~2 minutes once at the start of training, but can speed up the full run by 35% on A100 vs `_grouped_mm`, which is well worth it for longer runs.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252F5COXvwLZwdY61BhFvjnI%252Funknown.png%3Falt%3Dmedia%26token%3D59f07dcc-cb11-47d4-bacc-ec27e7454f19&width=768&dpr=3&quality=100&sign=7ab63621a2c74184cf4ad3dd02881e12&sv=3){width=590 height=390}

The larger the model and more context you use, **the more pronounced the memory savings from our Unsloth kernels will be** (efficiency will scale exponentially).

### 🧭 Automatic backend selection {#automatic-backend-selection}

Our main innovation is our **Split LoRA approach** for efficient MoE, which uses \~35% less memory and is 2x faster training when compared to Transformers v5 \+ `torch._grouped_mm`. Custom `torch._grouped_mm` \+ our Triton kernels are \~12-30x faster than Transformers v4.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FKHbKsPvtiK06Uogklven%252Fnicer_training_time_vs_batch_final_tweaks6_bold.png%3Falt%3Dmedia%26token%3D448853ff-b760-46ad-8e9b-599c6862762b&width=768&dpr=3&quality=100&sign=76ba42bdd8c04145f6f0c119221db2ec&sv=3){width=1904 height=864}

Training MoE models in **4-bit** QLoRA isn’t recommended right now because BitsandBytes doesn’t support it. This isn’t specific to Unsloth. For now, use bf16 for LoRA or full fine-tuning.

Unsloth will auto select either the following backends depending on your hardware:

You can also toggle them yourself:

To enable faster MoE training, update Unsloth via `pip install --upgrade unsloth unsloth_zoo`

### ❓What is torch.\_grouped\_mm? {#what-is-torch._grouped_mm}

Previously, Mixture-of-Experts (MoE) weights were stored as a `ModuleList` of per‑expert linear layers. The only practical way to run a forward pass was a for‑loop over experts, which is expensive and suboptimal.

PyTorch recently introduced [`grouped_mm`](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grouped_mm.html) to address this exact bottleneck. In parallel, we provide our own MoE‑optimized Triton kernels. This also lines up with a key Transformers change: as of Transformers v5, expert weights are stored as a [`single nn.Parameter`](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L226), making `grouped_mm` a natural fit for faster MoE training and inference.

So [transformers 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L222) changed:

to [transformers 5.0.0](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L226) style:

`torch._grouped_mm` works on GPUs starting with the NVIDIA T4, and we’ve verified it on H100, A100, B200, and RTX 6000 Pro, so support is broadly available.

We also previously introduced Unsloth [Flex Attention](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training) for gpt-oss, and these optimizations should make it even more efficient.

## 📊 Kernel Results \+ Benchmarks {#kernel-results--benchmarks}

Below is a comparison across sequence lengths for training speed and memory usage versus Transformers v5 (which already uses `torch._grouped_mm` for MoE). For **gpt-oss BF16 MoE training, we see 7x faster training and 36% VRAM reduction** on NVIDIA B200. For Qwen3-30B-A3B, it's 1.8x faster, and **GLM 4.7 Flash is 2.1x faster on RTX PRO 6000**. All benchmarks are done with LoRA rank \= 64 and all LoRA modules on MoE layers (gate, up, down).

### gpt-oss Benchmarks {#gpt-oss-benchmarks}

We fine-tuned [unsloth/gpt-oss-20b-BF16](https://huggingface.co/unsloth/gpt-oss-20b-BF16) for benchmarking. Unsloth is 7x faster and uses 36% less VRAM at 16K context lengths. Transformers v5 \+ TRL goes out of memory whilst Unsloth does not. Also the speed up increases with sequence length in this case thanks to our [Unsloth's Flex Attention implementation](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training#unsloths-flex-attention-implementation), and our MoE kernels.

### Qwen3 Benchmarks {#qwen3-benchmarks}

On an **NVIDIA B200**, we see **\~1.7x speedup and \~35% better memory efficiency with Qwen3-30B-A3B LoRA**, with memory savings improving further at longer sequence lengths.

Qwen3-Next and Coder surprisingly fit on a single B200 GPU in bf16 LoRA.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FonHXIql0XhGkLIDnuTfv%252Fimage.png%3Falt%3Dmedia%26token%3D06f97769-1d0e-4edb-b9c5-6c305376c6e8&width=768&dpr=3&quality=100&sign=1d186f98ef5191260fa6631a988a3e37&sv=3){width=4170 height=2175}

On H100 GPU, we perform significantly better than the baseline getting up to **1.77x speed up** in training while also saving \~5.3GB when fine tuning at 4K context length. While we seamlessly scale to 8192 context lengths, Transformers v5 \+ TRL OOMs at 8K. Notice that we use less memory at 8K than the baseline does at 4K so we can keep pushing the context length further.

### GLM 4.7 Benchmarks {#glm-4.7-benchmarks}

Unsloth achieves **2.6x faster throughput with >15% less VRAM** across all batch sizes for GLM 4.7 Flash. GLM 4.7 Flash is a 30B MoE (3B active parameters) agentic & coding model and employs a configuration similar to the DeepSeek MoE style, featuring 64 routed experts and 1 shared expert. We benchmarked Unsloth MoE training vs the new optimized Transformers v5.

Use our new Colab notebook for GLM 4.7 Flash below:

GLM 4.7 Flash MoE Notebook A100 80GB

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FbocwTistkljiMxGUA02y%252Fimage.png%3Falt%3Dmedia%26token%3D2a804b3a-44e4-4666-aa4f-ba6a31f69f39&width=768&dpr=3&quality=100&sign=15346cc5f83e83160a2a3a7659aeca3b&sv=3){width=4167 height=2149}

### ⚡Faster LoRA MoE training {#faster-lora-moe-training}

In Transformers/PEFT, the usual approach is to **merge the LoRA adapter into the base weight** and then run the MoE computation (especially since MoE often uses `nn.Parameter` instead of `nn.Linear`). The problem is that this merge effectively **materializes the LoRA delta (for all the experts)** `lora_B @ lora_A.t`, which is **very memory-hungry**.

Unsloth avoids that. We previously used the same idea to optimize generic LoRA training and inference, and we’ve now applied it to **MoE \+ LoRA** as well. The math is identical, so the loss, gradients, and outputs stay the same. The only change is **the order of operations**, made possible by matrix-multiplication associativity. With this reordering, we get major speedups and memory reductions.

Training MoE models in **4-bit** QLoRA isn’t recommended right now because BitsandBytes doesn’t support it. This isn’t specific to Unsloth. For now, use bf16 for LoRA or full fine-tuning.

These optimizations are **enabled by default** when training MoE models with Unsloth (notably Qwen-3 MoE, gpt-oss, and the models mentioned above). You can switch implementations via the `UNSLOTH_MOE_BACKEND` environment variable: either `torch._grouped_mm` **Triton kernels** or a **basic PyTorch for-loop**, depending on compatibility and preference. We default to `grouped_mm` for the best performance and broad support.

## 📚 Details of implementation {#details-of-implementation}

LoRA is a parameter-efficient fine-tuning method: instead of updating the full weight matrix, you train a low-rank “adapter” with far fewer parameters, which drastically reduces optimizer memory.

If the original weight has shape **(m, n)**, LoRA adds two trainable matrices with shapes **(m, r)** and **(r, n)**. Their product is **(m, n)**, but you only track optimizer states and gradients for:

- `m*r + r*n` parameters (LoRA) instead of
- `m*n` parameters (full fine-tuning)

For typical MLP layers, `m ≈ 4096, n ≈ 12k, and r ≈ 64`, that’s roughly **\~1M LoRA parameters vs \~48M full parameters -** about **\~2%,** often with minimal to no accuracy loss.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FogBmpbd8eAirmDJaDCMl%252FLoRA%2520Image%3Falt%3Dmedia%26token%3Dce533f2c-ad6a-4cca-8588-fe0f548f8bae&width=768&dpr=3&quality=100&sign=2c59de53733d7ad485036dabe7d42f2c&sv=3){width=340 height=299}

#### MoE LoRA changes things {#moe-lora-changes-things}

MoE layers are different because you have **E expert MLPs in parallel**, so any per‑expert change (like adding LoRA) scales across all experts.

Take **Qwen3‑30B‑A3B**: hidden size **m\=2048**, intermediate size **n\=768**, **E\=128** experts with **k\=8** activated per token. Per expert:

- `gate_proj` and `up_proj`: `(m, n) = (2048, 768)`
- `down_proj`: `(n, m) = (768, 2048)`

With **LoRA rank r\=64**, each projection adds `r*(m+n)=64*(2048+768)=180,224` parameters per expert (≈ `11%` of a `2048×768` matrix). The core issue is that `r/n = 64/768` is large compared to typical MLP setups, for e.g., `r/n = 64/25600` in [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json#L13) of similar size.

If you materialize this across *all* experts, memory adds up quickly. And since `gate_proj` and `up_proj` are often fused as `gate_up_proj`, you typically materialize both together, roughly doubling the overhead/peak memory.

**In terms of memory, for a sequence length s, E experts and**  `k`  **chosen, we have the following common for both approaches**

This is where things start to diverge. For peft’s approach we have

For Unsloth’s split LoRA approach, we perform the following operations

Now lets take the case of Qwen3-30B-A3B.

`E = 128, k = 8, m = 2048, n = 768.`Plugging all these in , we get `s < 32K.`

$$
\begin{matrix}\text{PEFT params} & :\quadEmn \\ \text{Unsloth Split LoRA params} & :\quadks(r+n) \\ \text{In typical LoRA we have} & :\quadr\lln \\ \text{Split LoRA is better when} & :\quadEmn>ksn\text{}=\text{}Em>ks \\  \\ \text{For Qwen3-30B-A3B, we have} \\ E & =128,\quadk=8,\quadm=2048,\quadn=768 \\  \\ \text{So, Split LoRA is mathematically better when} \\ s & <\frac{Emn}{kn}=32K\end{matrix}
$$

**In terms of compute, for a sequence length**  `s` **,**  `E`  **experts and top**  `k`  **chosen, we're doing:**

$$
\begin{matrix}\Delta=AB,A\in\mathbb{R}^{m\timesr},\text{}B\in\mathbb{R}^{r\timesn} & \quad\Rightarrow\quad2mnr\text{flops per expert lora} \\  \\ W^{′}=W+\Delta\quad & \Rightarrow\quadmn\text{flops} \\  \\ XW^{′}\quad∣\quadX\in\mathbb{R}^{s\timesm},\text{}W^{′}\in\mathbb{R}^{m\timesn}\quad & \Rightarrow\quad2smn\text{flops} \\  \\ \text{MoE peft lora flops} & =E(2mnr+mn)+2k\text{}smn\end{matrix}
$$

In case of Unsloth split lora that we mentioned, we have

$$
\begin{matrix}XW & =2smn\text{flops} \\ Y=XA, & =2smr\quad\text{(applied only to routed token–expert pairs)} \\ \text{}Z=YB & =2srn \\ \text{MoE split lora flops} & =2k(smn+smr+srn) \\ \text{Crossover condition} & :\quad2ksr(m+n)>2Emn(r+1/2)\Rightarrows>\frac{Emn}{k(m+n)}\times(1+\frac{1}{2r}) \\  \\ \text{For Qwen3-30B-A3B with} & :E=128,\text{}m=2048,\text{}n=768,\text{}k=8 \\  \\ \Rightarrow\quads & \text{}\approx\text{}16\text{K tokens}\end{matrix}
$$

The point till where the Split LoRA from analytical perspective is better is when `s > Emn/k(m+n)` which is in the order of `16K` tokens for Qwen3-30B-A3B style model.

Finally, some speedups come from **reduced memory traffic**: modern GPUs are often **bandwidth‑bound**, so transferring less data can matter more than FLOPs. A rough speedup estimate is `Emn / [k·s·(m+n)]`, so it depends strongly on **s, E, k**, and the matrix shapes.

### 🔮 Model Support {#model-support}

Unsloth supports faster MoE training for Qwen, gpt-oss, DeepSeek and GLM models:

- **Qwen3** (Thinking and Instruct): VL • 2507 • Coder 
- **gpt-oss**: 20B • 120B • safeguard
- **GLM**: 4.5 • 4.6 • 4.6-Air • 4.7 • 4.7-Flash
- **DeepSeek**: V3 • R1 • V3.1 • V3.2

We may have not uploaded some MoE models, but Unsloth should still support them.

### 📈 More Benchmarks {#more-benchmarks}

#### gpt-oss BF16 Benchmarks {#gpt-oss-bf16-benchmarks}

Training Speed including vs Transformers v4

**Memory VRAM usage**

## 🎉 Important Unsloth Updates {#important-unsloth-updates}

1. As part of our MoE release, we also made **Gemma-3 now use Flex-Attention** by default, and this works in float16 settings as well (there were infinities which we solved a while back). **Gemma-3 now uses O(N) memory and not O(N\^2) memory, and trains >3x faster** (scales even better with context length). Previous Unsloth versions would OOM.

![](https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FZiM9zMhVlUaJXC4Y1REp%252Fimage.png%3Falt%3Dmedia%26token%3Db2f1d12e-ccdb-431c-9b65-284db3892e2c&width=768&dpr=3&quality=100&sign=5fea348dd2932306a548fe3c3af4d4e0&sv=3){width=986 height=640}

1. Vision fine-tuning now accepts mixed data of only images and text data!
2. `trl==0.27.1` and `transformers==5.1.0` are supported well - previous coverage was 30% of all our 120 notebooks, but now we have >80% coverage - we plan to make it 100% over the next few days.

To enable faster MoE training, update Unsloth via `pip install --upgrade unsloth unsloth_zoo`

### Acknowledgements {#acknowledgements}

We thank the Hugging Face team for collaborating with us on improving MoE training for the community.

We also sincerely thank the torchao team, especially Vasily Kuznetsov (vkuzo) for working helping us enabling grouped\_mm support for float16 to get it work on T4 and backward compatibility with A100.
