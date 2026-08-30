Model

LLaMA-3 70B
LLaMA-2 13B
Gemma 7B
DeepSeek-V3
GLM-5.2
DeepSeek-V4-Pro
Qwen3.8-2.4T-A95B
Inkling
MiniMax-M3
custom…
D=
F=
L=
E=·k=·shared=
Hardware

TPU v5p
TPU v5e
H100 (8-GPU node)
B200 (8-GPU node)
GB200 NVL72
GB300 NVL72
H800 (DeepSeek)
custom…
C=
Wici=

specmeasured
Batch
B= tokens
reset

[What Do We Mean By Scaling?](#scaling)
[First, Feel the Roofline](#roofline)
[Data Parallelism](#data-parallelism)
[Fully-Sharded DP (FSDP)](#fsdp)
[Tensor Parallelism](#tensor-parallelism)
[FSDP + Tensor Parallelism](#mixed)
[Expert Parallelism](#expert-parallelism)
[Pipelining](#pipelining)
[Scaling Across Pods](#pods)
[The GPU Network Model](#gpus)
[Takeaways](#takeaways)
[Problems to Work](#problems)
[Appendix](#appendix)

# How to Parallelize a Transformer for Training

An *explorable* adaptation of [Part 5 of “How to Scale Your Model”](https://jax-ml.github.io/scaling-book/training/) by Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh, Albert Webson & Reiner Pope (Google DeepMind).

✦ We begin with the original dense TPU schemes — data parallelism, FSDP, tensor parallelism, their mixed form, and pipelining — then splice in the GPU fabric model and expert parallelism for MoEs. For each, we ask when communication becomes the bottleneck. (This summary is the adaptation’s; the chapter’s own dek described its four dense schemes.)

**This page is a working model, not a description of one.**
Every green number can be dragged left or right, or double-clicked to type an exact value. Every blue number is computed live from the green ones — try it here: drag the batch
 and watch the per-chip batch
 follow (hover any blue number for its formula). They share one model-and-hardware state, so a change made anywhere propagates everywhere. Parallelism degrees remain scheme-local: the dense mixed group uses N = DP·TP, while EP and PP are modeled in their own sections; composite worked examples state their full product explicitly. And scrub without fear —
reset restores every scrubbed number to its default while keeping your model, hardware, and spec/measured picks (it's the same button as in the top bar, which lights up orange whenever a scrub has strayed), any single number reverts on its own when you double-click it and commit it blank, and the browser's back button walks through your earlier configurations.

**Whose words are you reading?** Source passages come from the original TPU and GPU chapters (© 2022 Maruan Al-Shedivat, © 2025 Google LLC, [MIT license](LICENSE-scaling-book.txt)); AI-authored departures are explicitly labeled, with these conventions:
wherever the chapter printed a fixed number, this page computes it live (these in-place swaps aren't individually marked);
the interactive figures and their captions replace the original static figures;
✦ margin notes and passages explicitly labeled as adaptation are AI-written editorial voice — the initial edition was built by Fable (Anthropic) and this adversarial review and its corrections were performed by OpenAI Codex — including instructions, asides, and the new
[roofline primer](#roofline); the [expert parallelism](#expert-parallelism) and [GPU network](#gpus) sections instead mash up Chapter 12 source passages, with their AI-written connective and adaptation prose labeled by the same convention;
the chapter's single-letter mesh-axis names are rendered as named parallelism degrees throughout — its X is DP, its Y is TP, the pipelining section's Z is PP, and chapter 12's expert axis Z is EP (a global substitution; each is its own scrubbable variable, adjusted in the text where its section uses it);
under a GPU preset the hardware vocabulary follows suit — TPU→GPU, ICI→NVLink, DCN→InfiniBand, pod→node, MXU→tensor core — so the article reads as one consistent machine, and any TPU preset restores the chapter's exact words (sentences that deliberately *compare* the two never swap);
content woven into the chapter's text by this edition carries a dotted underline like that (and splice edits beside it that standard quotation practice would allow — a bracket or an ellipsis — go unmarked);
where a sentence had to be altered to host a live element, a Δ margin note quotes the original and states the change;
and where this edition's additions make a chapter statement inaccurate as written, an italic (Ed: …) interjection corrects it in place.

## What Do We Mean By Scaling?

The goal of “model scaling” is to be able to increase the number of
chips used for training or inference while achieving a proportional, linear
increase in throughput (we call this *strong scaling*). While performance
on a single chip depends on the trade-off between memory bandwidth and FLOPs,
performance at the cluster level depends on hiding inter-chip communication by
overlapping it with useful FLOPs. This is non-trivial, because increasing the
number of chips increases the communication load while reducing the amount of
per-device computation we can use to hide it. As we saw in
[Section 3](https://jax-ml.github.io/scaling-book/sharding/), sharded
matrix multiplications often require expensive
AllGathers or
ReduceScatters that can block the TPUs from doing
useful work. The goal of this section is to find out when these become
*too expensive.*

In this section, we'll discuss five common parallelism schemes: (pure)
**data parallelism, fully-sharded data parallelism** (FSDP / ZeRO
sharding), **tensor parallelism** (also known as model parallelism),
**expert parallelism** (for Mixture-of-Experts models),
and (briefly) **pipeline parallelism**. For each, we'll show what
communication cost we incur and at what point that cost starts to bottleneck our
compute cost.◦We'll
focus on communication bounds — since while memory capacity constraints are
important, they typically do not bound us when using rematerialization
(activation checkpointing) and a very large number of chips during pre-training.
(Ed: This edition is expanded to discuss
[expert parallelism](#expert-parallelism), unlike the
original.) For this section, you can focus solely on inter-chip
communication costs, since as long as we have a large enough single-chip batch
size, the transfer of data from HBM to MXU is already overlapped with
computation.

We'll use the following notation to simplify calculations throughout this
section.

live values shown for:
(mirrors the top bar)

| Notation | Meaning (model parameters) | Live value |
| --- | --- | --- |
| *D* | **d**model (the hidden dimension/residual stream dim) |  |
| *F* | **d**ff (the feed-forward dimension)✦adaptation **F convention (everywhere):** the width of *one expert* (= dff when dense); math runs through k·*F*, weights hold E·*F*, and the chapter's equations are the E = k = 1 case (Chapter 12's resolution). One honest limitation: models that mix dense and MoE blocks have *two* genuinely different F's — DeepSeek-V3 runs its first three layers dense at a much wider width — and this page approximates such models as uniformly MoE. Hover any *F* for the live widths. |  |
| *B* | Batch dimension (number of tokens in the batch; total, not per-device) |  |
| T | Sequence length | — |
| *L* | Number of layers in the model |  |

| Notation | Meaning (hardware characteristic) | Live value |
| --- | --- | --- |
| *C* | FLOPS/s per chip |  |
| *W* | Network bandwidth (bidirectional per TPU mesh axisone-way GPU or node egress, often subscripted as e.g. *Wici* or *Wdcn*) | ici ·  dcn |
| *DP* | Number of chips along the data-parallel mesh axis (the chapter's X) |  |
| *TP* | Number of chips along an alternate, tensor-parallel mesh axis (the chapter's Y) |  |
| *Z* | Number of chips along a third mesh axis, labeled Z | — |
| *PP* | Pipeline stages (the pipelining section's Z) |  |
| *EP* | Expert-parallel degree (chapter 12's Z; see the expert-parallelism section) |  |

✦ adaptation — this notation, worn by the chapter's dense models and today's frontier open-source ones (supported rows are clickable)

The chapter's examples are dense LLaMA-era models; the frontier has since gone
Mixture-of-Experts.✦Shapes
from each model's published `config.json` on Hugging Face; parameter
totals from its safetensors metadata. Retrieved August 2026.
E and k count shared experts, so k·*F* is the activated
width for the architectures represented by the live presets; column headers
explain each field. The dense models from the
top-bar dropdown lead the table for contrast, and whichever model is loaded
shows its row in live green — scrub it right here.

| Model | params | D | F | act. k·F | L | E | k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LLaMA-3 70B (chapter default) | 70.6B | 8,192 | 28,672 | 28,672 | 80 | 1 | 1 |
| LLaMA-2 13B | 13.0B | 5,120 | 13,824 | 13,824 | 40 | 1 | 1 |
| Gemma 7B | 8.54B | 3,072 | 24,576 | 24,576 | 28 | 1 | 1 |
| DeepSeek-V3✦Counting example: 256 routed + 1 shared expert → E 257; top-8 + shared → k 9. Its first three layers are actually dense (see the F-convention note above). | 685B | 7,168 | 2,048 | 18,432 | 61 | 257 | 8+1 |
| Kimi K3 (reference only)✦K3 is not a live preset because its routed experts operate after a projection from residual D = 7,168 into a 3,584-wide latent space. Its routed-expert intermediate width is F = 3,072. The page's single D×F expert model cannot represent both dimensions faithfully. | 2.78T | 7,168 | 3,072 | 55,296 | 93 | 896+2 | 16+2 |
| GLM-5.2 | 753B | 6,144 | 2,048 | 18,432 | 78 | 257 | 8+1 |
| DeepSeek-V4-Pro | 1.60T | 7,168 | 3,072 | 21,504 | 61 | 385 | 6+1 |
| Qwen3.8-2.4T-A95B | 2.45T | 8,192 | 2,048 | 22,528 | 92 | 513 | 10+1 |
| Inkling | 952B | 6,144 | 3,072 | 24,576 | 66 | 258 | 6+2 |
| MiniMax-M3 | 427B | 6,144 | 3,072 | 15,360 | 60 | 129 | 4+1 |

Click a supported model to load its shape (D, F, L, E, k) into the page's shared state
(the top bar follows); click a column header to sort. F = per-expert width;
act. k·F = activated width per token; E / k = total / activated experts,
counting shared. Across the supported live MoE presets, per-expert F is just
2,048 or 3,072, and activated width k·F clusters between 15k and 25k even as
total parameter counts span hundreds of billions to trillions. Since the
tensor-parallelism bound later in this chapter scales with the activated width
k·F, that clustering is why the TP limits look so similar across the supported
frontier presets. K3 is retained as a reference row, but its latent-MoE shape is
deliberately not loaded into these formulas.

✦ adaptation — the hardware, with receipts (click a row to load it)

Every hardware number this page computes with, spec and sustained, with its
source.✦Full
citations live in [`SOURCES.md`](SOURCES.md) alongside this
page — every value traces to a vendor spec sheet, a published measurement, or the
book's own benchmarks; retrieved 2026-08-17; click any cell to pin its citation
and follow the source link. Methodology for the synthesized numbers: NVIDIA
datasheets headline *sparsity* FLOP/s, halved here to dense; "bidirectional"
bandwidths are halved to per-direction; per-GPU scale-out is the node's NIC total
divided by its GPUs. ≈ marks factors that are *estimates* (stated basis, no
direct public measurement — e.g. Blackwell collectives inherit H100's measured
NCCL ratio until independent nccl-tests exist) rather than measured.
Hover any cell for its citation. The *sustained* and *achieved*
columns are the measured fractions of spec: flip the top bar's
**spec / measured** control and every equation on the page derates
by them (compute × sustained, bandwidth × achieved — the loaded hardware's row
shows them as live green scrubs). Wall-clock estimates that already assume an
MFU keep using spec peak, so nothing double-counts. Notice the punchline the
citations force: TPUs sustain far closer to their paper numbers than the
power-throttled NVIDIA parts.

| Hardware | C (dense bf16) | × sust. | W link | × achv. | W scale-out | HBM |
| --- | --- | --- | --- | --- | --- | --- |
| TPU v5p | 459 TF | ≈0.72 | 180 GB/s | ≈0.95 | 6.25 GB/s | 96 GB |
| TPU v5e | 197 TF | ≈0.67 | 90 GB/s | ≈0.95 | 3.13 GB/s | 16 GB |
| H100 (8-GPU node) | 989 TF | 0.73 | 450 GB/s | 0.82 | 50 GB/s | 80 GB |
| B200 (8-GPU node) | 2.25 PF | 0.69 | 900 GB/s | ≈0.82 | 50 GB/s | 180 GB |
| GB200 NVL72 | 2.5 PF | ≈0.70 | 900 GB/s | ≈0.82 | ≈50 GB/s | 186 GB |
| GB300 NVL72 | 2.5 PF | ≈0.70 | 900 GB/s | ≈0.82 | 100 GB/s | 288 GB |
| H800 (DeepSeek) | 989 TF | ≈0.73 | 200 GB/s | 0.80 | 50 GB/s | 80 GB |

For simplicity's sake, **we'll approximate a Transformer as a stack of
MLP blocks** — attention is a comparatively small fraction of the FLOPs
for larger models as we saw in
[Section 4](https://jax-ml.github.io/scaling-book/transformers/).
We will also ignore the gating matmul, leaving us with the following simple
structure for each layer:✦adaptation
With this simplification each layer holds
2·*D*·E·*F* weights (E = 1 for a
dense model, so simply 2·D·F), and the whole stack has
2·*D*·E·*F*·*L* =
 parameters at the current
settings — the “P” in this page's communication arithmetic. Memory
questions are different: a real checkpoint holds the gated MLP's third matrix
and the attention stack too, so the memory meters price weights at
Pw ≈ 3·D·E·F·L + 2.5·D²·L =
, which tracks the
model table's published totals to within a few percent (vocab embeddings and
MHA-era attention excepted).

A simplified Transformer layer. We treat each FFW block as a stack
of two matrices **Win**: bf16[D, F]
(up-projection) and **Wout**: bf16[F, D]
(down-projection) with an input **In**: bf16[B, D].
Box edges are drawn to (log) scale from the live dimensions — drag
*F* =
and watch the matrices fatten. Hover an edge to highlight that dimension
everywhere on the page.

Here's the full algorithm for our little Transformer with no parallelism.

**Forward pass:** need to compute Loss[B]

1. Tmp[B, F] = In[B, D] ·D Win[D, F]
2. Out[B, D] = Tmp[B, F] ·F Wout[F, D]
3. Loss[B] = …

**Backward pass:** need to compute dWout[F, D], dWin[D, F]

1. dOut[B, D] = …
2. dWout[F, D] = Tmp[B, F] ·B dOut[B, D]
3. dTmp[B, F] = dOut[B, D] ·D Wout[F, D]
4. dWin[D, F] = In[B, D] ·B dTmp[B, F]
5. dIn[B, D] = dTmp[B, F] ·F Win[D, F] (*needed for previous layers*)

We provide this for comparison to the algorithms with communication added.

Here are the 4 parallelism schemes we will discuss. Each scheme can be thought
of as uniquely defined by a sharding for **In**,
**Win, Wout, and Out** in the above
diagram.✦adaptation
A quick reminder of the book's notation: a subscript on an array dimension names
the mesh axis it is split over — In[BDP, D]
means the batch dimension is carved into *DP* pieces, one per
chip along axis *DP* — while a subscript on a · names the dimension being
contracted. The explorer below the four descriptions lets you click through the
schemes: for each one it draws all four arrays with their shards colored by
chip, the per-chip local shapes at the live *DP* and
*TP*, and the collectives the scheme pays for in the forward
and backward pass.

**1. Data parallelism:** *activations sharded along batch,
parameters and optimizer state are replicated on each device. Communication only
occurs during the backwards pass.*

In[BDP, D] ·D
Win[D, F] ·F
Wout[F, D] →
Out[BDP, D]

**2. Fully-sharded data parallelism (FSDP or ZeRO-3):**
*activations sharded along batch (like pure data parallelism), parameters
sharded along same mesh axis and AllGathered
just-in-time before use in forward pass. Optimizer state also sharded along
batch. Reduces duplicated memory.*

In[BDP, D] ·D
Win[DDP, F] ·F
Wout[F, DDP] →
Out[BDP, D]

**3. Tensor parallelism (also called Megatron sharding or model
parallelism):** *activations sharded along D (dmodel),
parameters sharded along F (dff).
AllGather and
ReduceScatter activations before and after each
block. Compatible with FSDP.*

In[B, DTP] ·D
Win[D, FTP] ·F
Wout[FTP, D] →
Out[B, DTP]

**4. Pipeline parallelism:** *weights sharded along the layer
dimension, activations microbatched and rolled along the layer dimension.
Communication between pipeline stages is minimal (just moving activations over a
single hop). To abuse notation:*✦adaptation
Notice what all four schemes have in common: every one runs the *same*
matmuls — the FLOPs never change, only where the arrays live and which
collectives must run between the multiplies. So for each scheme the question is
always whether those collectives can hide behind the matmuls. Before the chapter
dissects the schemes one by one, this adaptation inserts a short primer —
[First, Feel the Roofline](#roofline) — building the one picture that
answers that question for all four.

In[LPP, B, D][i] ·D
Win[LPP, D, F][i] ·F
Wout[LPP, F, D][i] →
Out[LPP, B, D][i]

The four schemes, one tab each. For the active scheme: its sharding
syntax line, the four arrays drawn with shards colored by chip, the per-chip
local shapes at the live
*DP* =  and
*TP* = ,
and the communication ops incurred in the forward and backward pass.

## First, Feel the Roofline

✦ This entire section is an addition of this adaptation — the chapter's ideas, our framing. The original text resumes at [Data Parallelism](#data-parallelism).

The core DP/FSDP/TP rooflines in this chapter are one picture wearing a few
costumes. Mixed sharding combines those clocks; expert and pipeline parallelism
add topology- and scheduling-specific activation traffic. Before we meet them,
let's get the core picture into your fingers.

When a chip works on one layer of our Transformer, two clocks run *at the same time*:

The **compute clock**: the MXU has to chew through this layer's share of FLOPs. With
B =  tokens
split over DP =  chips,
that's 4 · *B* · *D* · k·*F**DP* · *C* =
 per layer (k·F because a token only multiplies through its k activated experts).

The **network clock**: whatever bytes this scheme moves have to squeeze through the interconnect at
Wici = .
Crucially, these two clocks *can* overlap when the implementation schedules
the collective successfully: the network carries bytes while the MXU
multiplies. Under that explicit assumption, a layer costs the **max**,
not the sum. Communication that fits under the compute clock is hidden;
communication that pokes out past it leaves silicon idle.

The two clocks for one layer, drawn to scale from the numbers above. Drag B or DP in the paragraph (or anywhere on this page) and watch the bars. Flip the toggle to change *what* the network is carrying.

Try: with the toggle on **weights**, drag B =  down and watch only the *compute* bar shrink — the network bar doesn't hear the batch size at all, so at some point the comms must poke out. Then flip to **activations** and drag again: now both bars move together, and no amount of batch will save you.

That toggle captures the core DP/FSDP/TP distinction: *what travels?*

* If what travels is **weights** — 2·*D*·E·*F* bytes (every one of the E experts, active or not), as in data parallelism and FSDP — the network clock is *fixed*, while the compute clock scales with your per-chip batch. Big batch ⇒ compute wins ⇒ comms hidden.
* If what travels is **activations** — 2·*B*·*D* bytes, as in tensor parallelism — batch size appears on *both* clocks and cancels. Hiding then depends on the model's shape (the width a token multiplies through: k·*F*) instead.

Now the roofline itself. In [Part 1 of the original book](https://jax-ml.github.io/scaling-book/roofline/), a single chip was compute-bound only when its *arithmetic intensity* — FLOPs per byte touched — beat the ratio of FLOP speed to memory bandwidth. The identical logic applies here, one level up, with the interconnect playing the role of memory. For weight-moving schemes, your FLOPs scale with *B*/*DP* and your bytes don't, so:

*B**DP*
**is the dense-model token proxy for network arithmetic intensity**,
and for this equal-width MoE model the proportional intensity is
(*B*/*DP*)·k/E,
and the ridge of the roof sits at
Ek · *C**Wcollective*
=  tokens per chip — the E/k factor because a MoE moves all E experts' weights while its FLOPs only touch k.

The network roofline. Blue: weight-moving schemes climb the slanted part of the roof until the ridge, after which comms are fully hidden. Orange: tensor parallelism sits at a batch-independent height set by *F*, *TP*, and the hardware. The dot is *this page's* live operating point — dragging it drags B everywhere.

Try: drag the dot up the slope and feel where the ridge is. Then make the interconnect worse — drag Wici =  down — and watch the ridge slide right: a slower network demands a bigger per-chip batch before it can hide. Faster chips (drag C =  up) do the same thing, which is why each hardware generation makes parallelism *harder*, not easier.

**The recurring question:** *do this
scheme's bytes fit under this scheme's FLOPs?* For the core rooflines, the
answer compares a per-chip batch or model dimension with
*C*/*Wcollective*.
Expert and pipeline parallelism use the same instinct, with their own traffic and
scheduling caveats.

## Data Parallelism

*DP* =
chips share the batch of *B* =
tokens, so each model copy sees
 — drag either and
everything below follows.

**Syntax:**

In[BDP, D] ·D
Win[D, F] ·F
Wout[F, D] →
Out[BDP, D]

When your model fits on a single chip with even a tiny batch size (>240
tokens, so as to be compute-bound), **you should always use simple data
parallelism.** Pure data parallelism splits our activations across any
number of TPUs so long as the number of TPUs is smaller than our batch size.
The forward pass involves no communication, but at the end of every step,
**each TPU performs an AllReduce on its
local gradients to synchronize them before updating the
parameters.**

A diagram of pure data parallelism (forward pass). Our activations
(left) are fully sharded along the batch dimension and our weights are fully
replicated, so each TPU has an identical copy of the weights. This means the
total memory of our weights is increased by a factor of
*DP*, but no communication is required on the forward-pass.
Hover a chip to see exactly which slice of the batch it owns.

Here's the full algorithm for the forward and backwards pass. We abuse notation to write dL/dOut as dOut, purely for compactness.

**Pure Data Parallelism Algorithm:**

**Forward pass:** need to compute Loss[BDP]

1. Tmp[BDP, F] = In[BDP, D] ·D Win[D, F]
2. Out[BDP, D] = Tmp[BDP, F] ·F Wout[F, D]
3. Loss[BDP] = …

**Backward pass:** need to compute
dWout[F, D],
dWin[D, F]✦adaptation
The {UDP} annotation below marks a
result that is *unreduced* over the *DP* axis: each chip holds a
partial sum from its own slice of the batch.

1. dOut[BDP, D] = …
2. dWout[F, D] {UDP} = Tmp[BDP, F] ·B dOut[BDP, D]
3. dWout[F, D] = AllReduce(dWout[F, D] {UDP}) *(not on critical path, can be done async)*
4. dTmp[BDP, F] = dOut[BDP, D] ·D Wout[F, D]
5. dWin[D, F] {UDP} = In[BDP, D] ·B dTmp[BDP, F]
6. dWin[D, F] = AllReduce(dWin[D, F] {UDP}) *(not on critical path, can be done async)*
7. dIn[BDP, D] = dTmp[BDP, F] ·F Win[D, F] *(needed for previous layers)*

We ignore the details of the loss function and abbreviate
Tmp = Win · In. Note that, although our
final loss is the average
AllReduce(Loss[BDP]),
we only need to compute the AllReduce on the backward pass when averaging
weight gradients.

Note that the forward pass has no communication — **it's all in the
backward pass**! The backward pass also has the great property that the
AllReduces aren't in the “critical path”, meaning that each
AllReduce can be performed whenever it's convenient and doesn't block you from
performing subsequent operations. The overall communication cost *can still
bottleneck us* if it exceeds our total compute cost, but it is much more
forgiving from an implementation standpoint. We'll see that model/tensor
parallelism doesn't have this
property.✦adaptation
In [the primer](#roofline)'s terms: because the AllReduce can be
launched whenever convenient, the only question left is whether the total
comms fits under the total compute — the roofline question, answered just
below. Tensor parallelism's collectives, by contrast, block the very next
matmul.

**Why do this?** Pure data parallelism reduces activation
memory pressure by splitting our activations over the batch dimension, allowing
us to almost arbitrarily increase batch size as long as we have more chips to
split the batch dimension over. Especially during training when our activations
often dominate our memory usage, this is very helpful.

**Why not do this?** Pure data parallelism does nothing to
reduce memory pressure from model parameters or optimizer states, which means
pure data parallelism is rarely useful for interesting models at scale where
our parameters + optimizer state don't fit in a single TPU. To give a sense of
scale, if we train with parameters in bf16 and optimizer state in fp32 with
Adam◦Adam
stores parameters, first order and second order accumulators. Since the params
are in bfloat16 and optimizer state is in float32, this gives us
`2 + 8 = 10` bytes per parameters., the largest model
we can fit has TPU memory / 10 parameters, so e.g. on
a TPUv5p chip with
of HBM and pure data parallelism this is about
 parameters.

Per-chip HBM under pure data parallelism. Parameters and optimizer
state are replicated (they don't shrink as you add chips); only the
activations divide by *DP*. Drag *D*,
*F*, or *L* anywhere on the page and
watch the fixed 10-bytes-per-parameter slab crash through the HBM
ceiling.

**Takeaway**: the largest model we can train
with Adam and pure data parallelism has
num\_params = HBM per device / 10. For TPU v5p this is
roughly
parameters.◦Note
that this doesn't include gradient checkpoints, so this wouldn't actually be
useful. This is an absolute lower bound with a batch of 1 token.

*To make this useful for real models during training, we'll need to at
least partly shard the model parameters or optimizer.*

**When do we become bottlenecked by communication?** As we can
see above, we have two AllReduces per layer, each of size
2*D**F* (for
bf16 weights). When does data parallelism make us communication
bound?✦adaptation
The network here carries weight *gradients* —
2 · *D* · E · *F* =
 per matrix (all E experts' gradients, not just the k a token used) —
whose size doesn't care about the batch. This is the weight-moving roofline
from [the primer](#roofline): a batch-blind comms cost that a big
enough per-chip batch can always hide.

As in the table above, let *C* = per-chip FLOPs,
*Wici* = **bidirectional per-axis** ICI bandwidth**one-way per-GPU** NVLink egress,
bandwidth, and *DP* = number of shards across which the batch
is partitioned◦We
assume this partitioning is done over an ICI mesh, so the relevant network
bandwidth is Wici.. Let's calculate the time required
to perform the relevant matmuls, Tmath, and the required
communication time Tcomms. Since this parallelism scheme requires no
communication in the forward pass, we only need to calculate these quantities
for the backwards pass.

*Communication time:* From a previous section we know that the time
required to perform an AllReduce in a 1D mesh depends only on the total bytes
of the array being AllReduced and the ICI bandwidth
*Wici*; specifically the AllReduce time is
2 · total bytes / Wici. Since we need to AllReduce for both
Win and Wout, we have 2 AllReduces per layer. Each
AllReduce is for a weight matrix, i.e. an array of
*D**F* parameters, or
2*D**F* bytes. Putting this all together,
the total time for the AllReduce in a single layer is

Tcomms =
2 · 2 · 2 · *D* · *F**Wici*

right now, spread over
*MDP* =
 mesh axes (see the
Note on multiple mesh axes below):
8 ·  ·
 ·  /
 effective
Wcollective =
 per
layer — and notice: no *B* anywhere.

**✦ Adaptation:** This DP collective spans
more than one NVLink domain, so the live clock uses scale-out-limited bandwidth
, not the faster
 local NVLink egress.

One of the two per-layer gradient reductions:
AllReduce of dWin, an array of
 (all E experts' gradients travel). Press play to
watch the ring pass partial sums; the readout is the
2 · bytes / Wcollective cost used in the
derivation.

*Matmul time:* Each layer comprises two matmuls in the forward pass,
or four matmuls in the backwards pass, each of which requires
2(*B*/*DP*)*D**F*
FLOPs. Thus, for a single layer in the backward pass, we have

Tmath =
2 · 2 · 2 · *B* · *D* · *F**DP* · *C*

right now:
 per
layer — this one scales with the per-chip batch
*B*/*DP* =
 tokens (FLOPs carry k·F, the activated width).

Since we overlap, the total time per layer is the max of these two
quantities:

T ≈ max(
8 · *B* · *D* · *F**DP* · *C*,
8 · *D* · *F**Wici*
)

T ≈ 8 · *D* · *F* · max(
*B**DP* · *C*,
1*Wici*
)

We become compute-bound when
Tmath/Tcomms > 1, or
when

*B**DP*
>
*C**Wici*

right now:
 tokens/chip vs a
ridge at (E/k) · C/Wcollective =
 →
(headroom )

The upshot is that, to remain compute-bound with data parallelism, we need
the per-device batch size *B*/*DP* to
exceed the ICI operational intensity,
*C*/*Wici*. This is ultimately
a consequence of the fact that the computation time scales with the per-device
batch size, while the communication time is independent of this quantity (since
we are transferring model weights). Note the resemblance of the
*B*/*DP* > *C*/*Wici*
condition to the single-device compute-bound rule B > 240; in that case as
well, the rule came from the fact that computation time scaled with batch size
while data-transfer size was (in the B ≪ F, D regime) independent of batch
size.✦adaptation
Try: drag *DP* =
up and watch the compute bar in the meter below shrink toward the frozen comms
bar — every doubling of chips halves Tmath and does nothing to
Tcomms. The verdict flips the moment the per-chip batch dips below
.adaptation
With a Mixture-of-Experts model loaded (*E* > 1 — click e.g.
DeepSeek-V3 in the intro's frontier-models table), a penalty paragraph from
Chapter 12 appears just below the meter: the gradients move all
*E* experts while the FLOPs touch only the activated width, which
inflates this per-chip floor by E/k =
.

The two clocks for one data-parallel layer, backward pass, to
scale. The verdict flips exactly where
*B*/*DP* crosses
*C*/*Wcollective* — times E/k with a MoE loaded, as derived just below.

### MoE models✦adaptation This subsection is drawn verbatim from Chapter 12 (GPUs) of the same book and merged here by this adaptation; it appears only while a Mixture-of-Experts model is loaded (*E* = > 1). Chapter 12 states it for GPUs — hence “per-GPU” and *Wcollective*, its name for whichever network carries the collective — but the derivation is hardware-neutral: the same E/k inflation applies on any fabric with *Wcollective* read appropriately (here, over the TPU mesh or GPU fabric alike, the live check selects the bandwidth that carries the current DP collective).

For a Mixture of Experts (MoE) model, where we have *E* experts
and *k* experts per token, this increases to

Tmath =
2 · 2 · 2 · *k* · *B**D**F**DP* · *C*

Tcomms =
2 · 2 · 2 · *E* · *D**F**Wcollective*adaptation
Rendered exactly as Chapter 12 writes them — and Chapter 12's
*F*, the *per-expert* width, is precisely
[this page's convention](#scaling), so these formulas are the
page's live formulas verbatim, with exact (not approximate) E and
k.

which inflates the per-GPU token batch size by a factor of
*E*/*k*, i.e.

*B**DP*
>
*E**k*
*C**Wcollective*

right now: E/k =
, so the per-chip floor
of  tokens
inflates to (E/k) · C/Wcollective =
 — vs the
current B/*DP* =
 tokens/chip →

For example, the new OpenAI OSS model with k=4 and
E=128, this increases to
32 · 2475 = 79,200 across nodes, a kind of ridiculously
high number.✦adaptation
Kept as the chapter's static example (its 2475 is the H100 cross-node ridge from
[the GPU section](#gpus)). At your current hardware and model, the
same computation reads (E/k) · C/Wcollective =
 tokens per
chip. Expert parallelism — sharding the experts themselves, so gradients stop
crossing the whole DP axis — is the standard escape; it gets
[its own section](#expert-parallelism) below.

Let's put in some real numbers to get a sense of scale. For TPUv5p,
`C` =
and `W` =
for 1D data parallelism over ICI, so **our batch size per chip must be at
least  to avoid
being
communication-bound**.✦adaptation
The famous 2,550 — the same constant [the primer](#roofline)
promised would keep reappearing. It's live here: change the hardware in the
machine bar and this floor moves with it. Since we can do data
parallelism over multiple axes, if we dedicate all three axes of a TPUv5p pod
to pure data parallelism, we 3x our bandwidth
*Wici* and can scale down to only
BS= per TPU or
tokens per batch per pod (of
 chips)!
**This tells us that it's fairly hard to become bottlenecked by pure data
parallelism!**

Try:
8,192 chips (≈ a pod), 3 axes, 7.6M batch (right at the ridge)
same, but only 1 axis
page defaults

**Note [context parallelism]:** Throughout this
section, *B* always refers to the total batch size **in
tokens**. Clearly, however, our batch is made up of many different
sequences, so how does this work? As far as the MLP is concerned,
**tokens are tokens**! It doesn't matter if they belong to the same
sequence or two different sequences. So we are more or less free to do data
parallelism over both the batch and sequence dimension: we call this context
parallelism or sequence parallelism, but you can think of it as simply being
another kind of data parallelism. Attention is trickier than the MLP since we
do some cross-sequence computation, but this can be handled by gathering KVs or
Qs during attention and carefully overlapping FLOPs and comms (typically using
something called “ring attention”). Throughout this section, we
will just ignore our sequence dimension entirely and assume some amount of
batch or sequence parallelism.

**Note on multiple mesh axes:** We should
quickly note how multiple axes affects the available bandwidth. When we use
multiple mesh axes for a given parallelism strategy, we get more bandwidth.

* **Definition:** *MDP*
  (*MTP*, MZ, etc.) is the number of
  hardware mesh axes that a given parallelism strategy spans.
* **Effect (bandwidth-bound):** Using M axes provides
  (≈ M times) aggregate link bandwidth, so collective time scales
  ∝ 1/MDP.✦adaptation
  This is why the live lines above divide Tcomms by
  MDP. Right now *MDP* =
  ,
  so the ridge sits at C/(Wici·MDP) =
   tokens per
  chip rather than
  . Try: on the
  first preset above, scrub MDP between 1 and 3 and watch the
  verdict swing from communication-bound to compute-bound — same chips, same
  batch, just more wires carrying the same
  AllReduce.✦adaptation
  On a switched GPU fabric there are no extra mesh axes to recruit: each GPU
  has a single pipe into the NVLink switch, so M is pinned to 1 and the
  per-GPU ridge stays at C/Wegress =
   tokens — see
  [the GPU section](#gpus).

## Fully-Sharded Data Parallelism (FSDP)

**Syntax:**

In[BDP, D] ·D
Win[DDP, F] ·F
Wout[F, DDP] →
Out[BDP, D]

Fully-sharded data parallelism (often called FSDP or
[ZeRO-sharding](https://arxiv.org/abs/1910.02054)) splits the
model optimizer states and weights across the data parallel shards and
efficiently gathers and scatters them as needed. **Compared to pure
data parallelism, FSDP drastically reduces per-device memory usage and saves
on backward pass FLOPs, with very minimal overhead.**

FSDP shards the contracting dimension of Win and the
output dimension of Wout along the data dimension — here
*DP* =  shards.
This reduces memory but (from
[Section 3](https://jax-ml.github.io/scaling-book/sharding/))
requires us to gather the weights for W before we perform the matmul. Note
that the activations (left) *are not sharded along the contracting
dimension*, which is what forces us to gather. **Note that our
weight optimizer state is likewise sharded along the contracting
dimension.** Hover a chip to see which shards it owns.

You'll remember (from
[Section 3](https://jax-ml.github.io/scaling-book/sharding/))
that an AllReduce can be decomposed into an
AllGather and a
ReduceScatter. This means that, instead of
doing the full gradient AllReduce for standard data parallelism, we can
shard the weights and optimizer states across chips,
AllGather them at each layer during the
forward pass and ReduceScatter across the
weights during the backward pass at no extra cost.

The decomposition, animated: the
AllReduce that DP runs on each
2*D**E**F* =
 gradient is
exactly a ReduceScatter plus an
AllGather. FSDP performs the same two halves —
it just schedules them in different passes.

Here's the full algorithm for FSDP.

**Fully-Sharded Data Parallelism (FSDP):**

**Forward pass:** need to compute Loss[BDP]

1. Win[D, F] =
   AllGather(Win[DDP, F])
   (not on critical path, can do it during previous layer)
2. Tmp[BDP, F] =
   In[BDP, D] ·D
   Win[D, F]
   (can throw away Win[D, F] now)
3. Wout[F, D] =
   AllGather(Wout[F, DDP])
   (not on critical path, can do it during previous layer)
4. Out[BDP, D] =
   Tmp[BDP, F] ·F
   Wout[F, D]
5. Loss[BDP] = …

**Backward pass:** need to compute dWout[F, DDP], dWin[DDP, F]

1. dOut[BDP, D] = …
2. dWout[F, D] {UDP} =
   Tmp[BDP, F] ·B
   dOut[BDP, D]
3. dWout[F, DDP] =
   ReduceScatter(dWout[F, D] {UDP})
   (not on critical path, can be done async)
4. Wout[F, D] =
   AllGather(Wout[F, DDP])
   (can be done ahead of time)
5. dTmp[BDP, F] =
   dOut[BDP, D] ·D
   Wout[F, D]
   (can throw away Wout[F, D] here)
6. dWin[D, F] {UDP} =
   In[BDP, D] ·B
   dTmp[BDP, F]
7. dWin[DDP, F] =
   ReduceScatter(dWin[D, F] {UDP})
   (not on critical path, can be done async)
8. Win[D, F] =
   AllGather(Win[DDP, F])
   (can be done ahead of time)
9. dIn[BDP, D] =
   dTmp[BDP, F] ·F
   Win[D, F]
   (needed for previous layers) (can throw away Win[D, F] here)

This is also called "ZeRO Sharding", from "Zero Redundancy Optimizer"
since we don't perform any unnecessary compute or store any unnecessary
state. ZeRO-{1,2,3} are used to refer to sharding the optimizer states,
gradients, and weights in this way, respectively. Since all have the same
communication cost◦Technically,
FSDP adds communication in the forward pass that pure DP doesn't have, but
this is in the same proportion as the backward pass so it should have no
effect on the comms roofline. The key here is that ZeRO-3 turns a
backward-pass AllReduce into an AllGather and a ReduceScatter, which have
the same total comms volume., we can basically always do
ZeRO-3 sharding, which shards the parameters, gradients, and optimizer
states across a set of devices.

**Why would we do this?** Standard data parallelism involves
a lot of duplicated work. Each TPU AllReduces
the full gradient, then updates the full optimizer state (identical work on
all TPUs), then updates the parameters (again, fully duplicated). For ZeRO
sharding (sharding the gradients/optimizer state), instead of an AllReduce,
you can ReduceScatter the gradients, update
only your shard of the optimizer state, update a shard of the parameters,
then AllGather the parameters as needed for
your forward pass.✦adaptation Try:
in the memory meter below, drag *DP* =
down toward 1 and watch the meter fill up and overflow — at
*DP* = 1 you're just pure DP on one chip's memory budget.
Every doubling of *DP* halves the parameter and optimizer
stripes.

Per-chip HBM under FSDP: parameters
(2·Pw/*DP* =
) and Adam
optimizer state (8·Pw/*DP* =
) are now
divided by *DP*. Compare the pure-DP memory meter in the
previous section: there the same two stripes sat at their full, unsharded
size on every chip — 10 bytes per parameter, capping pure DP at about
 parameters per
 chip no matter
how many chips you buy.

**When do we become bottlenecked by communication?** Our
relative FLOPs and comms costs are exactly the same as pure data
parallelism, since each AllReduce in the
backward pass has become an AllGather +
ReduceScatter. Recall that an AllReduce is
implemented as an AllGather and a ReduceScatter, each with half the cost.
Here we model the forward pass since it has the same FLOPs-to-comms ratio as
the backward pass:✦adaptation The
chapter writes these equations for one mesh axis. The live line beneath them
(and every meter on this page) spreads the collective over
*MDP* =
mesh axes — the MDP the chapter's takeaway below refers to — so
Tcomms is divided by MDP.

Tmath =
2 · 2 · *B* · *D* · *F**DP* · *C*

Tcomms =
2 · 2 · *D* · *F**Wici*

T ≈ max(
4 · *B* · *D* · *F**DP* · *C*,
4 · *D* · *F**Wici*
)
= 4 · *D* · *F* · max(
*B**DP* · *C*,
1*Wici*
)

right now, per layer, over MDP =
 mesh axes (MoE: FLOPs carry k·F, weight comms E·F):
Tmath =  vs
Tcomms =  →

**✦ Adaptation:** This FSDP collective
crosses NVLink domains, so Wcollective is the scale-out-limited
.

Therefore, as with pure data-parallelism, we are compute bound when
*B* / *DP* >
*C* / *Wcollective*, i.e.
when the per-device batch size *B*/*DP*
exceeds the collective fabric's operational intensity
*C*/*Wcollective*
( /
 =
 right now). This
is great for us, because it means if our per-device batch size is big enough
to be compute-bound for pure data-parallelism, we can — without worrying
about leaving the compute-bound regime — simply upgrade to FSDP, saving
ourselves a massive amount of parameter and optimizer state memory! Though
we did have to add communication to the forward pass, this cost is
immaterial since it just overlaps with forward-pass FLOPs.

The two clocks for one FSDP layer. Same verdict as the DP
meter in the previous section, with both bars half as long — the ratio,
and therefore the ridge, is untouched.

**Takeaway:** Both FSDP and pure Data
Parallelism become bandwidth bound when the batch size per device is less than
tokens on the fabric carrying the collective.Δedited The
chapter reads: “Both FSDP and pure Data Parallelism become
bandwidth bound on TPUv5 when the batch size per device is less than
2550 / MX, where MX is the number of mesh
axes.” — generalized so the live number selects the fabric that actually
carries the collective (TPU mesh axes or the GPU domain hierarchy) and picks up
the MoE E/k factor.

For example, borrowing only DeepSeek-V2's reported batch size as a
*dense-model thought experiment* (this calculation does not model its
expert parallelism), take a batch size of ~40M tokens.✦adaptation This qualifier is added because the source imports DeepSeek-V2's batch into a dense FSDP calculation; it does not model that MoE's expert parallelism.
**This would allow us to scale to roughly
 chips,
or around
TPUv5 pods, before we hit a bandwidth
limit.**✦adaptation Load
the DeepSeek scenario with the button below, then make the batch your own:
the mini-calculator that follows is an addition of this edition. With
*B* =
tokens, FSDP scales to *DPmax* chips before hitting the bandwidth
limit.

*DPmax* =
*B* · *MDP* · *k*α · *E*
=  chips
✦adaptation This
equation is not in the chapter — it just solves the chapter's bound
B/*DP* > α/MDP (with a MoE loaded, the exact form
B/*DP* > (E/k)·α/MDP) for *DP*, so you can scrub
*B* and read off the largest compute-bound chip
count.

Try:
DeepSeek-V2 batch as dense thought experiment: B = 40M, all 3 axes
back to page defaults

For LLaMA-3 70B, which was trained for approximately
(15e12 · 70e9 · 6) FLOPs, we could split a batch of
tokens over roughly *B* / (α / 3) =
 chips
(roughly
pods of  chips), each with
 FLOPs running at
peak FLOPs utilization (often called MFU), and **train it in
approximately**.
Not bad! But let's explore how we can do
better.✦adaptation The
chapter's numbers (16M tokens, 18,823 chips, 17 days) are one point of this
live sentence — the recipe button below restores them. Then drag
*B* and watch chips and wall-clock trade off: a bigger
batch rides the same ridge on more chips and finishes sooner, which is
exactly why the labs fight for every doubling of critical batch size. The
equation below, also an addition, shows the wall-clock
arithmetic.

total FLOPschips · *C* · MFU
=
✦adaptation Added
by this edition: the chapter's 17-day estimate, kept live — total FLOPs
divided by the aggregate delivered FLOP-rate of the
 chips
at the chosen MFU.

Try:
chapter's LLaMA-3 recipe: B = 16M, 3 axes, MFU 50%

**Note on critical batch size**: somewhat
unintuitively, we become more communication bottlenecked as our total batch
size decreases (with fixed chip
number).✦adaptation See
for yourself: drag *B* =
down and watch the verdict above flip to
.
The hunt described at the end of this note is where the rest of this chapter
goes. Data parallelism and FSDP let us scale to arbitrarily
many chips so long as we can keep increasing our batch size! However, in
practice, as our batch size increases, we tend to see diminishing returns in
training since our gradients become almost noise-free. We also sometimes see
training instability. Thus, the game of finding an optimal sharding scheme
in the "unlimited compute regime" often starts from a fixed batch size,
determined by scaling laws, and a known (large) number of chips, and then
aims to find a partitioning that allows us to fit that small batch size on
so many chips.

## Tensor Parallelism

**Syntax:**

In[B, DTP] ·D
Win[D, FTP] ·F
Wout[FTP, D] →
Out[B, DTP]

(we use *TP* to eventually combine with FSDP)

In a fully-sharded data-parallel AllReduce we
move the weights across chips. We can also shard the feedforward dimension of
the model and move the activations during the layer — this is called
“1D model parallelism” or Megatron sharding
([Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053)). This can
unlock a smaller efficient batch size per pod. The figure below shows an example of a single matrix
sharded in this way:

An example of basic tensor parallelism. Since we're only sharding
our activations over *TP* (unlike in FSDP where we shard
over *DP*), we replicate our activations over
*DP*. Using our standard syntax, this is
A[B, DTP] ·
B[D, FTP] →
C[B, FTP]. Because we're only sharding
over one of the contracting dimensions, we typically
AllGather the activations *A* before the
matmul. Hover a chip to see which shards it owns.

As noted, **In[B, DTP] ·D
Win[D, FTP] ·F
Wout[FTP, D] →
Out[B, DTP] means we have to gather our
activations before the first matmul. This is cheaper than ZeRO sharding when
the activations are smaller than the weights.**✦adaptation
Compare the two freights live, per layer in bf16: gathering activations moves
2 · *B* · *D* =
, gathering
weights moves 2 · *D* · *E* · *F* =
 — right now the
lighter freight is
.
Try: drag *B* =
down toward the per-group batches a mixed scheme would see (a few thousand
tokens) and watch the verdict flip in the activations’ favor.
This is typically true only with some amount of ZeRO sharding added (which
reduces the size of the gather). This is one of the reasons we tend to mix
ZeRO sharding and tensor parallelism.

Here's the algorithm for tensor parallelism!

**Tensor Parallelism:**✦adaptation
Watch the phrase *on critical path*. With pure data parallelism the
AllReduce happened after the loss was already
computed, so the network could grind away while the chips moved on. Here
the matmuls cannot start until the gathers finish — these collectives sit
squarely in the layer’s serial path.

**Forward pass:** need to compute Loss[B]

1. In[B, D] = AllGather(In[B, DTP]) *(on critical path)*
2. Tmp[B, FTP] = In[B, D] ·D Win[D, FTP] *(not sharded along contracting, so no comms)*
3. Out[B, D] {UTP} = Tmp[B, FTP] ·F Wout[FTP, D]
4. Out[B, DTP] = ReduceScatter(Out[B, D] {UTP}) *(on critical path)*
5. Loss[B] = …

**Backward pass:** need to compute
dWout[FTP, D],
dWin[D, FTP]

1. dOut[B, DTP] = …
2. dOut[B, D] = AllGather(dOut[B, DTP]) *(on critical path)*
3. dWout[FTP, D] = Tmp[B, FTP] ·B dOut[B, D]
4. dTmp[B, FTP] = dOut[B, D] ·D Wout[FTP, D] *(can throw away dOut[B, D] here)*
5. In[B, D] = AllGather(In[B, DTP]) *(this can be skipped by sharing with (1) from the forward pass)*
6. dWin[D, FTP] = In[B, D] ·B dTmp[B, FTP]
7. dIn[B, D] {UTP} = dTmp[B, FTP] ·F Win[D, FTP] *(needed for previous layers)*
8. dIn[B, DTP] = ReduceScatter(dIn[B, D] {UTP}) *(on critical path)*

One nice thing about tensor parallelism is that it interacts nicely with
the two matrices in our Transformer forward pass. Naively, we would do an
AllReduce after each of the two matrices. But
here we first do **In[B, DTP] ·
Win[D, FTP] →
Tmp[B, FTP]** and then
**Tmp[B, FTP] ·
Wout[FTP, D] →
Out[B, DTP]**. This means we
AllGather **In** at the beginning,
and ReduceScatter **Out** at the
end, rather than doing an AllReduce.✦adaptation
And since an AllReduce is itself an AllGather
plus a ReduceScatter, one AllGather in and one ReduceScatter out is
*half* the bytes of the naive two-AllReduce plan.

**How costly is this?** Let's only model the forward pass -
the backwards pass is just the transpose of each operation here. In 1D tensor
parallelism we AllGather the activations before the first matmul, and
ReduceScatter them after the second, sending two bytes at a time (bf16). Let's
figure out when we're bottlenecked by communication.

Tmath =
4 · *B* · *D* · *F**TP* · *C*

Tcomms =
2 · 2 · (*B* · *D*)*Wici*

T ≈ max
4 · *B* · *D* · *F**TP* · *C*,
2 · 2 · (*B* · *D*)*Wici*

right now, per layer, with
*MTP* =
ICI axes carrying *TP* (MoE: the sharded math runs through the activated width k·F):
 of math vs
 of comms →

Noting that we want compute cost to be greater than comms cost, we
get:✦adaptation
Notice that *B* · *D* appears in
*both* clocks, so the batch cancels out of the ratio. This is the flat
orange line from [the primer](#roofline): tensor parallelism’s
compute-to-comms ratio is pinned at
*k* · *F* / (*TP* · αTP) =
 no matter
the batch — a weight-moving scheme can hide its comms behind more tokens per
chip, but no batch size can raise this bar.

4 · *B* · *D* · *F**TP* · *C*
>
2 · 2 · (*B* · *D*)*Wici*

*F**TP* · *C*
>
1*Wici*

*F* > *TP* ·
*C**Wici*

right now:
*F* =
(activated width k·F =
 — the width the
sharded math actually runs through)
vs *TP* · αTP =
at *TP* =
→

**✦ Adaptation:** This TP collective
spans NVLink domains. The live clock therefore uses
 scale-out-limited
bandwidth rather than the local .

Thus for instance, for TPUv5p,
*C*/*Wici* =
 in bf16, so we can
only do tensor parallelism up to
*TP* < *F* / .
When we have multiple ICI axes, our Tcomms is reduced by a factor of
*MTP*, so we get
*TP* < *MTP* · *F* / .

Tmath vs Tcomms for one tensor-parallel
layer, to scale. Unlike the DP and FSDP meters, dragging
*B* stretches *both* bars in lockstep — only
*F*, *TP*, and
*MTP* can change the verdict.

**Takeaway:** Tensor Parallelism becomes
communication bound when
*TP* · αTP > *F*
(or *k*·*F* for the page's equal-width MoE approximation).
For most models this is between 8 and 16-way tensor
parallelism.✦adaptation
On GPUs the live αTP changes when TP leaves the selected machine's
NVLink domain (8 GPUs on H100/B200, 72 on NVL72), which generally pins TP
inside one domain or at most two ([details](#gpus)).

**Note that this doesn't depend on the precision of the
computation**, since e.g. for int8, on TPUv5p,
Cint8/*Wici* is
 instead of
 but the comms
volume is also halved, so the two factors of two cancel.

**Let's think about some examples:**✦adaptation
The buttons below load each model's real shape into the page's state — every
number, meter, and verdict recomputes when you click one. Try: with a model
loaded, set *TP* =
to 8, then 16, then 32, and watch the verdicts. Or hold the model fixed and
scrub *C* =
:
faster chips shrink *TPmax* =
 on the
fabric carrying the current TP collective, which is
why each hardware generation makes tensor parallelism a little harder to
hide.

Load:
LLaMA-3 70B
Gemma 7B
chapter defaults

* On TPUv5p with LLaMA 3-70B with D = 8192,
  F ≈ 30,000, we can comfortably do 8-way tensor
  parallelism, but will be communication bound on 16 way tensor parallelism.
  The required F for 8-way model sharding is
  20k.✦adaptation
  Check it live at the current hardware: with LLaMA 3-70B's F = 28,672
  pinned, *TPmax* =
  -way;
  8-way is
  ,
  16-way is
  ,
  and the required F for 8-way sharding is 8 · α / *MTP* =
  .
* For Gemma 7B, this page's ignore-gating convention uses the config's
  F = 24,576. On TPU v5p that gives
  *TPmax* ≈ 9.64, so 8-way fits but 16-way is
  communication-bound.Δedited
  The chapter reads “For Gemma 7B, F ≈ 50k, so we become
  communication bound with 19-way tensor parallelism. That means we could likely
  do 16-way and still see good performance.” Its ≈50k adds Gemma's two
  gated input projections together, which is inconsistent with this chapter's
  two-main-matmul model. At the currently selected TPU, the corrected
  *TPmax* is
  -way,
  and 16-way is
  .

## Combining FSDP and Tensor Parallelism

**Syntax:**

In[BDP, DTP] ·D
Win[DDP, FTP] ·F
Wout[FTP, DDP] →
Out[BDP, DTP]

The nice thing about FSDP and tensor parallelism is that they can be
combined. By sharding **Win** and
**Wout** along both axes we both save memory and
compute. Because we shard B along *DP*, we reduce the size
of the model-parallel AllGathers, and because we shard
*F* along *TP*, we reduce the
communication overhead of FSDP. This means a combination of the two can get
us to an even lower effective batch size than we saw above.

A diagram combining FSDP and tensor parallelism. Unlike the
other cases, there is no duplication of model parameters: each of the
*N* =
chips holds a distinct
[D/*DP*, F/*TP*] =
[, ]
tile of Win, and the activations are sharded along both of their
axes too. Hover a chip to see which shards it owns.

Here's the full algorithm for mixed FSDP + tensor parallelism.
While we have a lot of communication, all our AllGathers and
ReduceScatters are smaller because we have batch-sharded our activations
and tensor sharded our weights much more!

**Forward pass:** need to compute
Loss[B]✦adaptation
Count what actually sits on the critical path: on the
*TP* axis, one AllGather
in (step 1) and one ReduceScatter out
(step 6) of activation bytes; the weight gathers on the
*DP* axis (steps 2 and 4) can be prefetched. Those two
*TP*-axis activation hops and two *DP*-axis weight hops are exactly the
2 · 2 factors in TTP comms and TFSDP comms
below.

1. In[BDP, D] = AllGatherTP(In[BDP, DTP]) *(on critical path)*
2. Win[D, FTP] = AllGatherDP(Win[DDP, FTP]) *(can be done ahead of time)*
3. Tmp[BDP, FTP] = In[BDP, D] ·D Win[D, FTP]
4. Wout[FTP, D] = AllGatherDP(Wout[FTP, DDP]) *(can be done ahead of time)*
5. Out[BDP, D] {UTP} = Tmp[BDP, FTP] ·F Wout[FTP, D]
6. Out[BDP, DTP] = ReduceScatterTP(Out[BDP, D] {UTP}) *(on critical path)*
7. Loss[BDP] = …

**Backward pass:** need to compute
dWout[FTP, DDP],
dWin[DDP, FTP]

1. dOut[BDP, DTP] = …
2. dOut[BDP, D] = AllGatherTP(dOut[BDP, DTP]) *(on critical path)*
3. dWout[FTP, D] {UDP} = Tmp[BDP, FTP] ·B dOut[BDP, D]
4. dWout[FTP, DDP] = ReduceScatterDP(dWout[FTP, D] {UDP})
5. Wout[FTP, D] = AllGatherDP(Wout[FTP, DDP]) *(can be done ahead of time)*
6. dTmp[BDP, FTP] = dOut[BDP, D] ·D Wout[FTP, D] *(can throw away dOut[B, D] here)*
7. In[BDP, D] = AllGatherTP(In[BDP, DTP]) *(not on critical path + this can be shared with (2) from the previous layer)*
8. dWin[D, FTP] {UDP} = In[BDP, D] ·B dTmp[BDP, FTP]
9. dWin[DDP, FTP] = ReduceScatterDP(dWin[D, FTP] {UDP})
10. Win[D, FTP] = AllGatherDP(Win[DDP, FTP]) *(can be done ahead of time)*
11. dIn[BDP, D] {UTP} = dTmp[BDP, FTP] ·F Win[D, FTP] *(needed for previous layers)*
12. dIn[BDP, DTP] = ReduceScatterTP(dIn[BDP, D] {UTP}) *(on critical path)*

**What's the right combination of FSDP and TP?** A simple but
key maxim is that FSDP moves weights and tensor parallelism moves
activations. That means as our batch size shrinks (especially as we do more
data parallelism), tensor parallelism becomes cheaper because our activations
per-shard are smaller.✦adaptation
This maxim is [the primer](#roofline)'s weights-move vs
activations-move toggle made load-bearing: per layer in bf16, FSDP's freight
is 2 · *D* · *E* · *F*/*TP* =
 of weights
while TP's is 2 · *B* · *D*/*DP* =
 of
activations — each scheme shrinks the *other's* bill.

* Tensor parallelism performs
  AllGatherTP([BDP, DTP])
  which shrinks as *DP* grows.
* FSDP performs
  AllGatherDP([DDP, FTP])
  which shrinks as *TP* grows.

Thus by combining both we can push our minimum batch size per replica down
even more. We can calculate the optimal amount of FSDP and TP in the same way
as above:

**TPU closed form.** Let *DP* be the number of chips dedicated to FSDP and
*TP* be the number of chips dedicated to tensor
parallelism. Let *N* be the total number of chips in our
slice with *N* = *DP**TP*.
Let *MDP* and *MTP*
be the number of mesh axes over which we do FSDP and TP respectively (these
should roughly sum to 3). We'll purely model the forward pass since it has
the most communication per FLOP. Then adding up the comms in the algorithm
above, we have

TFSDP comms(*B*, *DP*, *TP*) =
2 · 2 · *D* · *F**TP* · *Wici* · *MDP*

TTP comms(*B*, *DP*, *TP*) =
2 · 2 · *B* · *D**DP* · *Wici* · *MTP*

And likewise our total FLOPs time is

Tmath =
2 · 2 · *B* · *D* · *F**N* · *C*

right now, with
*MDP* =
mesh axes carrying FSDP and
*MTP* =
carrying TP (MoE: the weight gathers carry E·F, the FLOPs k·F):
TFSDP comms = ,
TTP comms = ,
Tmath =  per layer

**✦ Adaptation — GPU topology correction:**
the TPU equations immediately above are not valid GPU substitutions. An outer
FSDP reduction does not become TP times faster while TP remains inside one
NVLink domain; the scale-out link still carries the reduction. The live clocks,
meter, and explorers use Chapter 12's
max(Tdomain, Tscale-out) model.
In the equations below, bytes = 4·D·E·F and G is the selected NVLink-domain size.
The closed-form optimum below is therefore shown only on TPU; on GPU the
explorer finds the topology-aware minimum directly.

Tdomain =
bytes*WGPU egress*
· 1min(*TP*, G)

Tscale-out =
bytes*Wdomain egress*
· Gmax(G, *TP*)

TFSDP comms = max(Tdomain, Tscale-out)

To simplify the analysis, we make two assumptions: first, we allow
*DP* and *TP* to take on non-integer
values (as long as they are positive and satisfy
*DP**TP* = *N*);
second, we assume that we can fully overlap comms on the
*DP* and *TP* axis with each other. Under
the second assumption, the total comms time is

Tcomms = max(TFSDP comms, TTP comms)

Before we ask under what conditions we'll be compute-bound, let's find the
optimal values for *DP* and *TP* to
minimize our total communication. Since our FLOPs is independent of
*DP* and *TP*, the optimal settings are
those that simply minimize comms. To do this, let's write
Tcomms above in terms of *DP* and
*N* (which is held fixed, as it's the number of chips in
our system) rather than *DP* and *TP*:

Tcomms(*DP*) =
4 · *D**Wici*
· max(
*F* · *DP**N* · *MDP*,
*B**DP* · *MTP*
)

Because TFSDP comms is monotonically increasing in
*DP*, and TTP comms is monotonically decreasing
in *DP*, the maximum must be minimized when
TFSDP comms = TTP comms,
which occurs when

*F* · *DPopt**MDP*
= *B* · *N**DPopt* · *MTP*
 →
*DPopt* =
*B**F* · *MDP**MTP* · *N*

right now:
*DPopt* =  (the weight term is E·F wide, so E joins F under the radical)
→ nearest power of two:
chips of FSDP, with the current
*DP* = ,
*TP* =

This is super useful! This tells us, for a given *B*,
*F*, and *N*, what amount of FSDP is
optimal. Let's get a sense of scale. Plugging in realistic values, namely
*N* = 64 (corresponding to a 4x4x4 array of chips),
*B* = 48,000, *F* = 32768, gives
roughly *DP* ≈ .
So we would choose *DP* to be 16 and
*TP* to be 4, close to our calculated
optimum.✦adaptation
The chapter rounds this to ≈13.9; the pinned live value here is
√(48,000 · 2 · 64 / 32,768) exactly. And at whatever is loaded right now,
*DPopt* =
.
Press the first button below to load the chapter's exact scenario into the
whole page.

Try:
chapter example: 4×4×4 cube, B=48k, F=32,768
back to page defaults

**Takeaway:** in general, during training,
the optimal amount of FSDP is
*DPopt* =
*B**F* · *MDP**MTP* · *N*.✦adaptation
You can *feel* this optimum in the figure below, which holds
*N* fixed and sweeps the split. Try: drag the marker to
either extreme — all-FSDP on the right, all-TP on the left — and watch the
envelope climb; then drag
*B* =
and watch the valley migrate right like √*B*, or widen
*F* =
and watch it slide left.

Trading FSDP for tensor parallelism at fixed
*N* = .
Blue: TFSDP comms, rising in *DP*. Orange:
TTP comms, falling in *DP*. The emphasized
upper envelope is what you actually wait for; the dashed ink line is
Tmath, and wherever the envelope dips below it the layer is
compute-bound. Drag the marker: it sets *DP* (and
*TP* = *N*/*DP*) for
the whole page. The dotted drop-line marks
the TPU closed-form
*DPopt* =
the numerically best constrained continuous split under the GPU hierarchy.

Now let's return to the question we've been asking of all our parallelism
strategies: **under what conditions will we be
compute-bound?** Since we can overlap FLOPs and comms, we are
compute-bound when✦adaptation
Same question as [the primer](#roofline)'s: does the slower of the
two comms clocks fit under the compute clock?

max(TFSDP comms, TTP comms) < Tmath

By letting
α ≡ *C* / *Wici*,
the ICI arithmetic intensity, we can simplify:

max(
*F**TP* · *MDP*,
*B**DP* · *MTP*
)
<
*B* · *F**N* · α

Since we calculated *DPopt* to make the LHS maximum equal, we can
just plug it into either side (noting that
*TPopt* = *N*/*DPopt*),
i.e.

*F**N* · *Wici* · *MDP*
*B**F* · *MDP**MTP* · *N*
<
*B* · *F**N* · *C*

Further simplifying, we find that

*B* · *F**MDP* · *MTP* · *N*
<
*B* · *F**N* · α,

where the left-hand-side is proportional to the communication time and the
right-hand-side is proportional to the computation time. Note that while the
computation time scales linearly with the batch size (as it does regardless
of parallelism), the communication time scales as the square root of the
batch size. The ratio of the computation to communication time thus also
scales as the square root of the batch size:

TmathTcomms
=
*B**F* *MDP* *MTP*α *N*

right now:
Tmath/Tcomms =
at the optimal mix, with α =
 (MoE: the ratio gains k/√E — FLOPs carry k·F, the gathered weights E·F)

To ensure that this ratio is greater than one so we are compute bound, we
require

*B**N*
>
α2*MDP* · *MTP* · *F*

right now:
*B*/*N* =
 vs threshold
α²·E/(MDP·MTP·k²·F) =
tokens/chip →

To get approximate numbers, again plug in *F* = 32,768,
α = 2550, and
*MDP**MTP* = 2
(as it must be for a 3D mesh). This gives roughly
*B*/*N* >
.
This roughly wins us a factor of eight compared to the purely data parallel
(or FSDP) case, where assuming a 3D mesh we calculate that
*B*/*N* must exceed about
 to be compute
bound.✦adaptation
At this page's live values the threshold is
tokens/chip (the 850 for pure FSDP is α/3, all three mesh axes carrying the
weight collectives). Watch the two clocks race in the meter below. Try: drag
*B* =
down until the verdict flips to comms-bound, then check
*B*/*N* =
 against the
threshold — the flip lands right at the ridge (assuming
*DP* is near *DPopt*; if it's far off, the flip
comes sooner).

Tmath vs Tcomms = max(TFSDP,
TTP) for one mixed FSDP + TP layer, to scale, with the verdict
at the current *DP* =
,
*TP* =
.

**Takeaway:** combining tensor parallelism
with FSDP allows us to drop to a
*B*/*N* of
255022*F*.
This lets us handle a batch of as little as 100 per chip, which is roughly a
factor of eight smaller than we could achieve with just FSDP.

Below we plot the ratio of FLOPs to comms time for mixed FSDP + TP,
comparing it both to only tensor parallelism (TP) and only data parallelism
(FSDP), on a representative 4x4x4 chip array. While pure FSDP parallelism
dominates for very large batch sizes, in the regime where batch size over
number of chips is between roughly 100 and 850, a mixed FSDP + TP strategy is
required in order to be
compute-bound.✦adaptation
The live chart below plays this figure's role: flip its view toggle to
*ratio* to see Tmath/Tcomms for all three
schemes, where any curve above 1 is compute-bound. It is drawn at the page's
current *N* =
 chips — press the
4×4×4 chapter-example preset above to reproduce the chapter's exact
frame.

Here's another example of TPU v5p 16x16x16 showing the FLOPs and comms
time as a function of batch size for different sharding
schemes.✦adaptation
That second figure is the same chart in *absolute-times* view. A
16x16x16 slice is 4096 chips — exactly the page's default
*DP*·*TP* = 512 · 8, so the
“back to page defaults” preset above reproduces it.

The black curve is the amount of time spent on model FLOPs, meaning any
batch size where this is lower than all comms costs is strictly comms bound.
You'll notice the black curve intersects the aqua curve at about
4e5, as
predicted.✦adaptation
On the live chart that crossing sits at
*B* = *N*·α²·*E*/(*MDP*·*MTP*·*k*²·*F*) =
total tokens; the chapter's 4e5 is this same formula evaluated at its
16x16x16, F ≈ 30k scenario.

Here's an interactive animation to play with this, showing the total
compute time and communication time for different batch sizes:

Time per layer versus batch size for
*N* =
chips. Dashed ink: Tmath, the matmul FLOPs — any comms curve
sitting above it means that scheme is comms-bound at that batch. Blue:
FSDP-only comms, flat, because weights don't hear the batch size. Orange:
TP-only comms, rising in lock-step with Tmath. Aqua: the best
constrained continuous mixed FSDP + TP split, rising like √*B* under the piecewise GPU hierarchy. Drag the vertical
marker to set *B*; flip the view toggle between absolute
times and the Tmath/Tcomms ratio, where any curve
above 1 is compute-bound.

You'll notice this generally agrees with the above (minimum around
FSDP=256, TP=16), plus or minus some wiggle factor for some slight
differences in the number of axes for
each.✦adaptation
The chapter's animation swept the FSDP/TP split itself; on this page that
sweep is the earlier *DP*-axis explorer, whose optimum at the current state is
*DPopt* =
→ nearest power of two
-way
FSDP. The ±wiggle from mesh-axis bookkeeping is exactly the
MDP, MTP factors you can scrub above.

## Expert Parallelism

✦ This section is
drawn from [Chapter 12
(GPUs)](https://jax-ml.github.io/scaling-book/gpus/) of the same book and merged into this chapter's flow by this
adaptation; condensed passages are marked. Its cost model is Chapter 12's
switched GPU fabric (NVLink node + InfiniBand scale-out — see
[the GPU network model](#gpus) below). One naming change
throughout: Chapter 12 calls the expert-parallel axis *Z*; this
edition names every parallelism degree after its scheme, so that axis is
rendered *EP* here. The routed/shared-expert split and the
hardware-domain generalization of the H100-specific formula are AI-written
adaptation material, labeled again at the live estimate.

You currently have a **dense**
model loaded (*E* = ,
*k* = ), so
expert routing is not applicable. Click a Mixture-of-Experts preset — here, in the intro's
frontier-models table, or in the top bar — to make this section live:
DeepSeek-V3 (256 routed + 1 shared, 64-way EP)

You are on a **TPU** hardware
preset. This section's AllToAll cost model is Chapter 12's switched GPU fabric,
so its live verdicts are hidden until you select a GPU preset in the top bar
(or in [the GPU section](#gpus)).

As we've already noted above, Mixture of Expert (MoE) models come with
*E* times more model weights with only *k* times
more FLOPs, making data parallelism significantly
harder.✦adaptation In
Chapter 12 "noted above" pointed at its Data Parallelism section; on this page that
passage lives at the end of [Data Parallelism](#data-parallelism).
This page's *E* and *k* count all experts, including
always-on shared experts. Expert routing instead uses
Er = E − s routed experts and
kr = k − s routed selections, where
*s* =
shared experts. Right now, *E*r =
 and
*k*r = .
We can mitigate the routed weight cost by sharding along the expert dimension, i.e.
Win[EEP, D, F]. To do
the MLP block, we need to introduce 2x
AllToAll to send our activations to the
corresponding experts.

**What does an AllToAll cost here?** GPUs within a node have
all-to-all connectivity, which makes AllToAlls, well, quite easy: each GPU just
sends directly to the destination. For Mixture of Expert (MoE) models, we
frequently want to do a *sparse or ragged AllToAll*, where we guarantee at
most *k*r of *N* shards on the output dimension
are non-zero; the cost is reduced by
kr/N.✦adaptation Condensed
from Chapter 12's intra-node collectives discussion (two paragraphs on dense and
ragged AllToAlls, with the exact expected-occupancy footnote) — see Chapter 12 for
the full derivation. The takeaway below is carried verbatim.

**Takeaway:** The cost of an AllToAll on an
array of B bytes on GPU within a single node is about
Tcomms = (B · (8 − 1)) / (8² · *WGPU egress*)
≈ B / (8 · *WGPU egress*). For a ragged
(top-*k*r) AllToAll, this is decreased further to
(B · *k*r) / (64 · *WGPU egress*).✦adaptation On
a TPU mesh the equivalent dense cost is ≈ bytes/(4·*W*).
Chapter 12 makes the comparison itself: “Compare this to a
TPU, where the cost is $B / (4W)$. Thus, within a single node, we get a 2X
theoretical speedup in runtime ($B / 4W$ vs. $B / 8W$).”

For the eight-GPU H100 node used in Chapter 12, the cost of this
AllToAllEP→kr([B, D, k]) if it
spans multiple nodes is roughly
TAllToAll = 2 · *B* · *D* · (EP − 8)/(*W* · EP) · min(8 · *k*r / EP, 1),
so for pure expert parallelism we
needΔedited The
chapter reads: “the cost of this AllToAllZ->k([B, D, k])
if it spans multiple nodes is roughly $T\_\text{AllToAll} = 2 \cdot B \cdot D \cdot
(Z-8)/Z \min(8 \* k / Z, 1)$” — its inline expression omits the division by
*W*, evidently a typo (the chapter's own displayed
Tcomms just below includes it); restored here.

Tmath =
4 · *B* · *k* · *D* · *F*EP · *C*adaptation Chapter
12's expression verbatim for a routed-only MoE. The live expression below
keeps shared experts in total activated compute while excluding them from
routing traffic.

Tcomms =
4 · *B* · *D* · (EP − 8)*W* · EP
· min(
8 · *k*EP, 1
)

right now, with
*EP* =
-way
expert parallelism and a fast-fabric domain of
 GPUs:
Tmath =
 vs
Tcomms =
per layer →
not applicable to a dense model
invalid: require 1 ≤ kr and EP ≤ Er

**✦ Adaptation:** The live estimate is the substantive mash-up here. Within one
NVLink domain it uses the finite ragged AllToAll cost from Chapter 12 rather
than calling that transfer free. Beyond the domain it takes the slower of the
local switched-fabric component and the chapter's scale-out component, replacing
the H100-specific 8 with the selected hardware's domain size. Shared experts
remain in the total *k*-wide compute, but never become routed AllToAll
destinations.

For that H100 case, Chapter 12 concludes that we either need
kr > EP/8 with
*F* > α · (EP − 8)/kr
or EP ≫ kr and
*F* > 8 · α, where
α = *C*/*W*. This
gives you two domains in which expert parallelism is possible, one with a small
amount of expert parallelism (roughly 2-node) and small *F*,
or one with large *F* and EP arbitrarily large (up
to *E*r-way expert parallelism).

right now:
F =  vs
G · αdomain = G · C/Wdomain, for G =
, is
 →
;
at the current EP =
, the small-EP
condition F > α · (EP − G)/kr =
is

You'll see both cases in practice, either a small amount of expert-parallelism
(like DeepSeek v3 which has very small *F* and relatively
small, restricted cross-node expert parallelism), or models with large
*F*, in which case we can do significant cross-node EP
alongside TP.

**Takeaway:** if
*F* < G · *C* / *Wdomain*,
expert parallelism can span roughly 1–2 fast-fabric domains with similar
(slightly lower) cost to TP, or if
*F* > G · *C* / *Wdomain*,
we can do a significant amount of expert parallelism, up to
*E*r-way EP, with relatively low cost.

## Pipelining

You'll probably notice we've avoided talking about pipelining at all in the
previous sections. Pipelining is a dominant strategy for GPU parallelism that is
somewhat less essential on TPUs. Briefly, pipelined training involves splitting the
layers of a model across multiple devices and passing the activations between
pipeline stages during the forward and backward pass.✦adaptation On
this page the split is live: with *L* =
 layers over
*PP* =
pipeline stages (scrubbable below), each device owns about
 consecutive
layers. The algorithm is something like:

1. Initialize your data on TPU 0 with your weights sharded across the layer
   dimension (Win[LPP, DDP, FTP]
   for pipelining with FSDP and tensor parallelism).
2. Perform the first layer on TPU 0, then copy the resulting activations to
   TPU 1, and repeat until you get to the last TPU.
3. Compute the loss function and its derivative
   ∂L/∂xL.
4. For the last pipeline stage, compute the derivatives
   ∂L/∂WL and
   ∂L/∂xL−1, then copy
   ∂L/∂xL−1 to the previous pipeline stage
   and repeat until you reach TPU 0.

Here is some (working) Python pseudo-code

This pseudocode should run on a Cloud TPU VM. While it's not very efficient or
realistic, it gives you a sense how data is being propagated across devices.

```
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# Pretend each layer is just a single matmul.
x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model))

def layer_fn(x, weight):
  return x @ weight

# Assume we have num_layers == num_pipeline_stages
intermediates = [x]
for i in range(num_layers):
  x = layer_fn(x, weights[i])
  intermediates.append(x)

  if i != num_layers - 1:
    x = jax.device_put(x, jax.devices()[i+1])

def loss_fn(batch):
  return jnp.mean(batch ** 2)  # make up some fake loss function

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(num_layers - 1, -1, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i], weights[i])
  dx, dw = f_vjp(dx)  # compute the jvp dx @ J(L)(x[i], W[i])
  weights[i] = weights[i] - 0.01 * dw  # update our weights

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

**Why is this a good idea?** Pipelining is great for many reasons:
it has a low communication cost between pipeline stages, meaning you can train very
large models even with low bandwidth interconnects. This is often very useful on
GPUs since they are not densely connected by ICI in the way TPUs
are.✦adaptation The
chapter doesn't quantify "low communication cost," so the check below is ours. A
stage-boundary hop is a single point-to-point copy of one activation block —
2*D* =
per token in bf16 — and it's the [same roofline question](#roofline) as
ever: does the hop fit under one stage's compute clock? The line below runs the
numbers for one microbatch.

Thop = 2·*D*·(*B*/*Mmicro*) ÷ *Wstage* =
vs one stage's compute
 →
✦adaptation Try:
this check passes with room to spare — slash
*Wici* =
tenfold and it *still* passes, because the hop's cost has a factor of
*F*·layers-per-stage of compute to hide behind. That headroom
is the entire reason pipelining tolerates weak networks. On GPU this check
conservatively uses the slower of per-GPU NVLink and aggregate node egress,
assuming adjacent stages can cross a node boundary; on TPU it uses ICI. It
is a pure-PP boundary model: orthogonal DP/TP sharding would reduce the
per-rank activation block and requires an explicit stage layout.

**Why is this difficult/annoying?** You might have noticed in the
pseudocode above that TPU 0 is almost always idle! It's only doing work on the very
first and last step of the pipeline. The period of idleness is called a pipeline
bubble and is very annoying to deal with. Typically we try to mitigate this first
with microbatching, which sends
*Mmicro* =
small batches through the
*PP* =
-stage
pipeline, keeping TPU 0 utilized for at least a larger fraction of the total step
time.

bubble fraction =
*PP* − 1*Mmicro* + *PP* − 1
✦adaptation This GPipe-style bubble fraction is added by this edition; the source describes microbatching qualitatively but does not state the formula.

right now:
 /
 =
 idle →
per-device utilization ✦adaptation Try:
drag *Mmicro* down to 1 — the worst case, where each
device touches the batch exactly once and the bubble hits
 — then crank it
toward 64 and watch the hatched region in the schedule below shrink like
(*PP*−1)/(*Mmicro*+*PP*−1).

The overall communication cost of pipelining is tiny: with
*NMB* microbatches and
*Nstages*, we have
Tcomms per hop = 2 · *B* · *D* / (*W* · *NMB*)
and
*NMB* + *Nstages* − 2
hops, so
roughly✦adaptation This
passage — visible because a GPU preset is loaded — is Chapter 12's treatment of
pipeline parallelism, moved here into the chapter's pipelining section by this
adaptation. Its cost formula and reason (2) are verbatim; reasons (1) and (3)
are condensed to a sentence each, as marked.

Ttotal PP comms =
2*B**D**W* · *NMB*
· (*NMB* + *Nstages* − 2)

Tper-layer comms ≈ 1.5 ·
2*B**D**W* · *Nlayers*

right now, over effective stage-link bandwidth
 with
Nlayers = :
per-layer PP comms ≈
vs the DP AllReduce's
 per layer

Since we are dividing by *Nlayers*, this is vastly
smaller than any of the other costs. In other words, from a communication
standpoint, pipelining is basically free. So why don't we just do pipelining?
There are a few reasons:

(1) **Code complexity:** pipelining fits poorly into automatic
parallelism frameworks (like XLA's GSPMD), because microbatching and custom
zero-bubble schedules change the structure of the
program.✦adaptation Condensed
to one sentence — see Chapter 12 for the full paragraph.

(2) **Pipelining makes data parallelism and FSDP hard:** probably
the biggest reason not to do pipelining is that it plays badly with FSDP and data
parallelism. ZeRO-3 sharding in particular works badly, since it requires us to
AllGather the weights on every microbatch which
doesn't work when we have only
*B* / Nmicrobatches tokens
to amortize the AllGather cost. Furthermore, during the backward pass, *we
can't AllReduce or ReduceScatter the gradients until the last microbatch has
passed a given stage, which means we have significant non-overlapped
communication time.*

(3) **Pipeline bubbles and step imbalance:** naive pipeline
schedules leave stages idle in bubbles, and passing activations from stage to
stage on the critical path shifts stages relative to each other and adds
overhead.✦adaptation Condensed
to one sentence — see Chapter 12, and the live bubble math just above.

There are workarounds for each of these issues, but they tend to be
complicated to implement and difficult to maintain; pipelining remains a
technique with low communication cost relative to other methods.

A second approach is to carefully overlap the forward matmul
Wi @ xi, the backward
dx matmul
Wi @ ∂L/∂xi+1, and the
dW matmul
∂L/∂xi+1 @ xi. Since each of these
requires some FLOPs, we can overlap them to fully hide the bubble. Here's our live
stand-in for the plot from the recent
[DeepSeek v3 paper](https://arxiv.org/abs/2412.19437) showing their
"bubble-free" pipeline
schedule:✦adaptation Toggle
the widget's mode: *naive* is GPipe, *1F1B* interleaves one forward
with one backward (same bubble, far less activation memory held live), and
*overlap-dW* is the DeepSeek-v3-style schedule — rush every ∂x result down
the pipeline to unblock neighbors, and drop the deferred ∂W matmuls into slots that
would otherwise sit idle.

The pipeline schedule, live — rows are devices, time runs left to
right. Blue blocks are forward microbatches, orange is the backward ∂L/∂x work
(drawn 2× wide), aqua is the deferred ∂L/∂W work, and the hatched gaps are the
bubble. In *overlap-dW* mode — standing in for the figure from the
DeepSeek v3 paper — prioritizing the backwards ∂L/∂x multiplications avoids
"stranding" FLOPs.

Try:
no microbatching (worst case)
PP=4, M=8
deep pipeline, M ≫ PP

Because it is less critical for TPUs (which have larger interconnected pods), we
won't delve into this as deeply, but it's a good exercise to understand the key
pipelining bottlenecks.✦adaptation The
condensed picture: pipelining's communication is one activation hop per stage
boundary, so it thrives on weak interconnects and dominates GPU training; the price
is the bubble — currently
 of each
device's time — which microbatching shrinks and careful ∂x/∂W overlap can
erase.

## Scaling Across Pods

The largest possible TPU slice is a TPU v5p SuperPod with 8960 chips (and 2240
hosts). When we want to scale beyond this size, we need to cross the Data-Center
Networking (DCN) boundary. Each TPU host comes equipped with one or several NICs
(Network Interface Cards) that connect the host to other TPU v5p pods over Ethernet.
As noted in the [TPU
Section](https://jax-ml.github.io/scaling-book/tpus/), each host has about 200Gbps (25GB/s) of full-duplex DCN bandwidth,
which is about
full-duplex (egress) bandwidth per
TPU.✦adaptation Per
the TPU chapter of the original book: each v5p host serves 4 chips, so 25 GB/s per
host ÷ 4 ≈ 6.25 GB/s of egress per
chip.adaptation The
chapter printed 6.25GB/s; here *Wdcn* is scrubbable —
drag it and this whole section (ridge included) recomputes. The hardware presets in
the top bar set it per machine.

Typically, when scaling beyond a single pod, we do some form of model parallelism
or FSDP within the ICI domain, and then pure data parallelism across multiple pods.
Let *N* =
be the number of TPUs we want to scale to and
*M* =
be the number of TPUs per ICI-connected slice. To do an
AllReduce over DCN, we
can do a ring-reduction over the set of pods, giving us (in the backward pass):

Tmath =
2 · 2 · 2 · *B**D**F**N* · *C*

Tcomms =
2 · 2 · 2 · *D**F**M* · *Wdcn*

right now:
Tmath = .
All chips fit in one ICI domain, so no DCN collective is required.
Tcomms =  →
balanced per-pod batch  vs
ridge (E/k)·αdcn =  →
.

**✦ Adaptation:** The printed derivation assumes full, equal-size slices
(*N* is a multiple of *M*). The live model
balances the chips across
slices, so a partial final slice cannot silently receive a full slice's aggregate
NIC bandwidth.

The comms bandwidth scales with *M*, since unlike ICI the total
bandwidth grows as we grow our ICI domain and acquire more NICs. Simplifying, we
find that Tmath > Tcomms when

*B*slice
>
*C**Wdcn*

For TPU v5p, the
*C*/*Wdcn*
is about
 /
 =
. This tells us
that to efficiently scale over DCN, there is a minimum batch size per ICI domain
needed to egress each
node.✦adaptation This
is the [primer's roofline](#roofline) wearing one more costume — same
slanted roof, much worse network. The ridge that sat at
 tokens *per
chip* over ICI now sits at
 tokens *per
pod* over DCN (both carrying the MoE's
E/k weight-to-FLOPs factor).

Your current *N* =
 chips arranged as
 pod(s) of up to
, with DCN links between
them. Each pod runs FSDP + TP internally; the pods trade gradients over Ethernet.
Green means the per-pod batch clears the
-token ridge.

The two clocks for cross-pod data parallelism, per layer. The comms bar
is fixed by the model shape and the pod's aggregate NIC bandwidth; only the compute
bar hears
*B*.✦adaptation Try:
drag *B* =
down until the verdict flips — the comms bar never moves, exactly the
weight-moving pattern from the primer. Then shrink the slice
()
and watch Tcomms grow: fewer hosts per pod means fewer NICs sharing the
same gradient bytes.

**How much of a problem is this?** To take a specific example, say we
want to train LLaMA-3 70B on TPU v5p with a BS of
tokens. LLaMA-3 70B has *F* ≈
.
From the above sections, we know the following:

Load the scenario:
LLaMA-3 70B, BS = 2M

* We can do Tensor Parallelism up to
  *TP* = *MTP* · *k* · *F* /
  ≈  · *MTP*.✦adaptation *MTP* =
  here — drag it and the TP cap scales with the number of ICI axes devoted to
  *TP*.
* We can do FSDP so long as
  *B* / *N* >  / *MDP*.
  That means if we want to train with BS=
  and 3 axes of data parallelism, we'd at most be able to use ≈
   chips, roughly a
  quarter of a TPU v5p
  pod.✦adaptation With
  the page's live *MDP* =
  instead of the sentence's 3 axes, the cap is
   chips (the k/E factor is the MoE data-parallel penalty from [Data Parallelism](#data-parallelism)).
* When we combine FSDP + Tensor Parallelism, become comms-bound when we have
  *B* / *N* <
  ² · *E* / (*MDP*·*MTP* · *k*² · ) =
  ,
  so this lets us scale to roughly
   chips!
  However, the maximum size of a TPU v5p pod is 8k chips, so beyond that we have
  to use
  DCN.✦adaptation The
  chapter writes this threshold as α²/2F, taking MDP·MTP = 2;
  the live form uses the page's MDP·MTP (and, for a MoE, the
  E/k² factor from [the mixed derivation](#mixed)). Live
  check against the current slice size:
  .

The TLDR is that we have a nice recipe for training with BS=1M, using roughly
*DP* (FSDP) = 1024 and *TP* (TP) = 8, but with
BS=2M we need to use DCN. As noted above, we have a DCN arithmetic intensity of
, so we just need to
make sure our batch size per ICI domain is greater than this. This is trivial for
us, since with 2 pods we'd have a per-pod BS of
, and a per TPU
batch size of , which is
great (maybe cutting it a bit close, but theoretically
sound).✦adaptation The
chapter's printed values (per-pod BS of 1M, per-TPU batch of 111) appear when you
load the two-pod preset below; everything is recomputed from the live state, so try
the one-pod recipe first and watch both numbers move.

Try:
BS = 1M, one pod (*DP*=1024, *TP*=8)
BS = 2M, two pods

**Takeaway:** Scaling across multiple TPU pods is
fairly straightforward using pure data parallelism so long as our per-pod batch size
is at least  tokens.

## The GPU Network Model

✦ This section is an
addition of this adaptation, drawing its text from
[Chapter 12 (GPUs)](https://jax-ml.github.io/scaling-book/gpus/) of the
same book; condensed passages are marked. Chapter 12's per-scheme roofline
derivations are not repeated here — they re-derive what this chapter already
derived, so they are merged into the scheme sections above (the MoE penalty into
[Data Parallelism](#data-parallelism), the TP bound into
[Tensor Parallelism](#tensor-parallelism), expert parallelism into
[its own section](#expert-parallelism), and the pipelining reasons into
[Pipelining](#pipelining)). What remains here is the network model
itself: the fabric, its bandwidths, what collectives cost on it, and the worked
examples.

You are currently on a **TPU**
hardware preset, so the live numbers in this section follow *your* hardware,
not the H100s these sentences describe — click a GPU preset below (or in the top
bar) and the section's numbers snap to Chapter 12's published values.

Now let's look at what this has all been building towards: understanding
rooflines for LLM scaling on GPU. This is to complement the TPU training chapter
[here](#scaling). As we did there, the goal here is to look at the total
Tmath and Tcomms for different parallelism strategies and
understand at what point Tcomms > Tmath. As before, we
consider only the MLP block with operations

MLP(x) ≡ x[B, D] ·D
Win[D, F] ·F
Wout[F, D]

where *B* is the global batch size **in tokens**
(i.e. *B* = batch size · sequence length).

Here we'll reproduce the table from Chapter 12 showing effective bandwidths at
both the GPU and node level:

| Node Type | GPUs per node | GPU egress bandwidth | Node egress bandwidth |
| --- | --- | --- | --- |
| H100 | 8 | 450e9 | 400e9 |
| B200 | 8 | 900e9 | 400e9 |
| GB200 NVL72 | 72 | 900e9 | 3600e9 |
| GB300 NVL72✦adaptation This row is the adaptation's, not Chapter 12's — from NVIDIA's published GB300 NVL72 specs (dense BF16 = 180 PFLOPS/rack ÷ 72 = 2.5 PFLOP/s per GPU; ConnectX-8 at 800 Gb/s per GPU doubles the scale-out egress to 7200e9 per domain). | 72 | 900e9 | 7200e9 |

**Note:** Both the GPU and node egress bandwidths
determine rooflines for our LLMs. We'll use the term
*Wcollective* to describe either the GPU or node
bandwidths depending on whether we are operating within or above the node level.

Let's look at the compute communication rooflines as we did for TPUs for
**data parallelism, tensor parallelism, pipeline parallelism, expert
parallelism,** and combinations thereof. For the rest of this section we'll
focus on H100 rooflines for specific calculations. GB200-NVL72 has the same general
rooflines but because we have a larger node egress bandwidth, we can sometimes be
bottlenecked at the node level instead. The scheme derivations are merged into
their corresponding sections above; below are the bounds they land on here.

Here is the mapping used by the live GPU rooflines: read
*Wici* as the per-GPU **egress bandwidth**
into the NVLink switch fabric (450 GB/s on H100, 900 GB/s on B200); read a
**pod** as an NVLink domain (8 GPUs per node, or 72 on GB200 NVL72);
read *Wdcn* as each GPU's share of its node's
InfiniBand egress (400e9/8 = 50 GB/s on H100 and B200, and likewise 3600e9/72 on
GB200); and, because a switched fabric has no mesh axes, set
*MDP* = *MTP* = 1. Then
*C*/*Wici* =
 is the in-node ridge,
*C*/*Wdcn* =
 the per-GPU
scale-out ratio. DP/FSDP/TP select the local or scale-out collective bandwidth
from their degree. Mixed FSDP+TP needs one extra correction: Chapter 12's
hierarchical max(Tdomain, Tscale-out)
rule, because inner TP does not accelerate an outer reduction until it spans
more than one NVLink domain. The mixed meters and explorers implement that rule.

Hardware:
H100
B200
GB200 NVL72
GB300 NVL72
H800 (DeepSeek)
back to TPU v5p

### Where the bounds land on this fabric

For data parallelism and ZeRO sharding, the compute-bound rule derived in
[Data Parallelism](#data-parallelism) —
*B*/*DP* >
*C*/*Wcollective* — is
reused unchanged, where *Wcollective* is either
the GPU or node level egress bandwidth depending on whether we're sharding within a
node or across nodes. Thus:

* **Within a node**, we just need the per-GPU **token**
  batch size >
   /
   =
  .
* **Within an SU or at the spine level**, BS >
   /
   =
  .

This is quite a bit higher than on a TPU, where the number is 850 with all
three axes. On the H100 scale-out fabric the dense asymptotic floor is
990e12/400e9 = 2,475 tokens per GPU, so 16,384 GPUs would require about
40.6M tokens before the small-ring and model-parallel refinements; Llama 3.1
405B used 16M. Chapter 12 quoted a 3,300-token H800 baseline from an unsupported
300 GB/s figure. The reconciled H800 *local-link dense baseline* is 4,950
in spec mode (990e12/200e9) and about 4,517 in this page's measured mode. Those
are not a model of DeepSeek's full sparse run: its EP, PP, and 2-way DP alter the
outer reduction. DeepSeek reports a pretraining batch schedule from 3,072 to
15,360 sequences at a 4K maximum sequence length — about 12.6M to 62.9M tokens,
with 62.9M at steady state.Δedited
The source says H800 has 300 GB/s and “in practice, they used
4M”. H800 is 200 GB/s per direction by the reconciled spec, DeepSeek
reports 160 GB/s measured, and its report gives the sequence-batch schedule
above. See the [hardware table](#hardware-table).

**Small-DP correction.** The asymptotic ridge above omits the
ring factor. With *X* scale-out domains, the exact dense condition is
*B*/*N* >
(*C*/*Wcollective*) · (X−1)/X
(and ×E/k for the equal-width MoE model). For exactly two
domains the floor is halved, which is why 2-way data parallelism appears so
often.

✦ The same data-parallel meter from earlier on the page, mounted here
as adaptation chrome: flip the hardware presets above — H100, B200, GB200 NVL72,
H800, back to TPU v5p — and watch the identical two clocks answer for each fabric.
With *MDP* = 1 on a switched fabric, the verdict uses
the fabric carrying the current DP degree:
*B*/*DP* >
.

For tensor parallelism, the bound from
[Tensor Parallelism](#tensor-parallelism) —
*TP* < *F* · *Wcollective* / *C* —
gives about *F*/-way
TP within a node, and pipelining — whose
Chapter 12 treatment now lives in [Pipelining](#pipelining) under GPU
presets — is basically free from a communication standpoint. As with above, we
get an extra 2X bandwidth when we span exactly 2 fast-fabric domains. For the
chapter's eight-GPU H100 node this usually permits 16-way TP; generalized to a
domain of G = , the
refinement is
*F* >
 ·
(*TP* − G).

### What collectives cost on this fabric

**Takeaway:** the cost to AllGather or
ReduceScatter an array of B bytes within a single node is about
Tcomms = B · (8 − 1) / (8 · *WGPU egress*)
≈ B / *WGPU egress*. This is theoretically
around B / 450e9 on an H100 and B / 900e9 on a B200. An
AllReduce has 2x this cost unless in-network
reductions are enabled.

Beyond the node level: to a first approximation, because we have full bisection
bandwidth, the cost of an AllGather or
ReduceScatter is roughly the buffer size in bytes
divided by the node egress bandwidth (400GB/s on H100) *regardless of any of
the details of the tree reduction.*

TAG or RS comms =
bytes*Wnode egress*

right now: one layer's weights, 2 ·
*D* · *E* · *F* =
, cost
 to gather
in-node vs
across nodes

With in-network reductions enabled and using pure data parallelism, theoretically
we have 2x the AllReduce bandwidth, which would halve both of these numbers.
However, in practice the benefit is closer to 30%, which only really makes up for
the fact that we typically struggle to reach the reported numbers. Furthermore,
because pure data parallelism is rarely useful, this basically doesn't matter in
practice.

**Takeaway:** in theory, NVIDIA SHARP (available
on most NVIDIA switches) should reduce the cost of an
AllReduce on B bytes from about
2 · B / *W* to
B / *W*. However, in practice we only
see a roughly 30% improvement in bandwidth. Since pure AllReduces are fairly rare
in LLMs, this is not especially useful.

**Takeaway:** although NVIDIA claims bandwidths
of about 450GB/s over an H100 NVLink, it is difficult in practice to exceed 370
GB/s, so adjust the above estimates accordingly.

**Caveat about latency:** As noted before, GPUs
struggle to achieve full AllReduce bandwidth even with fairly large messages. This
means even if we in theory can scale e.g. expert-parallel AllToAlls across multiple
nodes, we may struggle to achieve even 50% of the total bandwidth. This means we do
try to keep TP or EP within a smaller number of nodes to minimize latency
overhead.

### Examples

**What does DeepSeek do?** For reference,
[DeepSeek V3](https://arxiv.org/abs/2412.19437) is trained with 2048
H800 GPUs with:✦adaptation These
two worked examples are Chapter 12's. The buttons load their cited hardware,
model shape, batch, and listed parallelism degrees. Because the page's generic
chip count is DP·TP, widgets that do not explicitly model EP or PP should be read
as component-level views, not as a reconstruction of the full training run.

* 64-way Expert Parallelism (EP) spanning 8 nodes
* 16-way Pipeline Parallelism (PP)
* 2-way ZeRO-1 Data Parallelism (DP)

They had a steady state batch size of
4096 · 15360 = 62,914,560 tokens, or 30k tokens per GPU.
You can see that this is already quite large, but their model is also very sparse
(k=8, E=256) so you need a
fairly large batch size. You can see that with 64-way EP and 16-way PP, we end up
with 1024-way model parallelism in total, which means the AllReduce is done at the
spine level, and because it's only 2-way, we end up with
2 / (2 − 1) = 2 times more bandwidth in practice. This
also helps reduce the cost of the final data-parallel AllReduce overlapping with
the final pipeline stages.

Load it:
DeepSeek V3 on 2048 H800s

**What does Llama 3.1 405B do?** Llama 3.1 405B trains with a BS of
16M tokens on 16,384 H100 GPUs, or about 977 tokens per GPU. They do:

* 8-way Tensor Parallelism within a node (TP)
* 16-way Pipeline Parallelism (PP)
* 128-way ZeRO-1 Data Parallelism

The decomposition is 8 TP · 16 PP · 128 DP = 16,384
GPUs. This is also a dense model so in general these things are pretty trivial. The
16-way PP reduces the cost of the data parallel AllReduce by 16x, which helps us
reduce the critical batch size.

Load it:
Llama 3.1 405B on 16,384 H100s

### TLDR of LLM scaling on GPUs✦adaptation Restored from Chapter 12's omitted “TLDR of LLM Scaling on GPUs” and practical recipe; condensed to avoid repeating the full derivations already merged into the scheme sections.

* DP or FSDP needs a local batch of roughly 2,500 dense-model tokens per
  GPU on H100-scale fabrics; MoEs multiply that floor by E/k, while a small
  DP degree benefits from the ring correction above.
* TP is usually compute-bound only within one NVLink domain, or at most
  roughly two. NVL72 expands that local domain but does not remove the
  topology check.
* Model parallelism that spans domains can reduce the outer FSDP cost, but
  on GPUs the reduction tracks *domains spanned*, not merely the inner
  sharding degree.
* PP has low byte cost if its scheduling complexity, bubbles, and delayed
  gradient reductions are handled successfully.

**Practical recipe:** smaller dense models can use aggressive
FSDP when batch permits; larger dense models commonly combine one- or two-domain
TP with many-stage PP and DP; MoEs add EP, generally preferring it to TP while
keeping latency-sensitive collectives within as few domains as practical.

## Takeaways from LLM Training on TPUs

* Increasing parallelism or reducing batch size both tend to make us more
  communication-bound because they reduce the amount of compute performed per
  chip.
* Up to a reasonable context length (~32k) we can get away with modeling a
  Transformer as a stack of MLP blocks and define each of several parallelism
  schemes by how they shard the two/three main matmuls per layer.
* During training there are 4 main parallelism schemes we consider, each of
  which has its own bandwidth and compute requirements (data parallelism, FSDP,
  tensor parallelism, and mixed FSDP + tensor parallelism):

| Strategy | Description |
| --- | --- |
| **Data Parallelism** | Activations are batch sharded, everything else is fully-replicated, we all-reduce gradients during the backward pass. |
| **FSDP** | Activations, weights, and optimizer are batch sharded, weights are gathered just before use, gradients are reduce-scattered. |
| **Tensor Parallelism (aka Megatron, Model)** | Activations are sharded along dmodel, weights are sharded along dff, activations are gathered before Win, the result reduce-scattered after Wout. |
| **Mixed FSDP + Tensor Parallelism** | Both of the above, where FSDP gathers the model sharded weights. |

And here are the "formulas" for each method:

| Strategy | Formula |
| --- | --- |
| DP | In[BDP, D] ·D Win[D, F] ·F Wout[F, D] → Out[BDP, D] |
| FSDP | In[BDP, D] ·D Win[DDP, F] ·F Wout[F, DDP] → Out[BDP, D] |
| TP | In[B, DTP] ·D Win[D, FTP] ·F Wout[FTP, D] → Out[B, DTP] |
| TP + FSDP | In[BDP, DTP] ·D Win[DDP, FTP] ·F Wout[FTP, DDP] → Out[BDP, DTP] |

* Each of these strategies has a limit at which it becomes
  network/communication bound, based on their per-device compute and comms.
  Here's compute and comms per-layer, assuming *DP* is FSDP and
  *TP* is tensor parallelism:

| Strategy | Compute per layer (ignoring gating einsum) | Comms per layer (bytes, forward + backward pass) |
| --- | --- | --- |
| DP | 4*B**D**F*/*DP* + 8*B**D**F*/*DP* | 0 + 8*D**F* |
| FSDP | 4*B**D**F*/*DP* + 8*B**D**F*/*DP* | 4*D**F* + 8*D**F* |
| TP | 4*B**D**F*/*TP* + 8*B**D**F*/*TP* | 4*B**D* + 4*B**D* |
| FSDP + TP | 4*B**D**F*/(*DP**TP*) + 8*B**D**F*/(*DP**TP*) | (4*B**D*/*DP* + 4*D**F*/*TP*) + (8*B**D*/*DP* + 8*D**F*/*TP*) |

* Pure data parallelism is rarely useful because the model and its optimizer
  state use bytes = 10x parameter count. This means we can rarely fit more than a
  few billion parameters in
  memory.✦adaptation Live:
  at  of HBM per chip,
  a fully replicated model + Adam state caps out around
   parameters.
* Data parallelism and FSDP become comms bound when the
  batch size per shard < *C* / *W*,
  the arithmetic intensity of the network. For ICI this is
   and for DCN this is
  about . This can
  be increased with more parallel axes.
* Tensor parallelism becomes comms bound when
  |*TP*| > *F* / .
  **This is around 8-16 way for most models.** This is independent of
  the batch
  size.✦adaptation At
  the current *F* that bound is
  -way per
  axis (k·F, the activated width, is
  what the sharded matmuls run through).
* Mixed FSDP + tensor parallelism allows us to drop the batch size to as low as
  ² / 2*F*
  ≈ . This
  is remarkably low.✦adaptation The
  live number uses the general floor α²·E/(MDP·MTP·k²·F) —
  the chapter's α²/2F is the dense case with two mesh axes.
* Data parallelism across pods requires a minimum batch size per pod of roughly
   before becoming
  DCN-bound.
* Basically, if your batch sizes are big or your model is small, things are
  simple. You can either do data parallelism or FSDP + data parallelism across
  DCN. The middle section is where things get interesting.

## Takeaways

**✦ Adaptation:** The source table below this
anchor is the dense TPU summary, so it is hidden for the current state rather
than allowed to display false MoE or GPU formulas. For GPU guidance, use the
[GPU TLDR and practical recipe](#gpus); for MoE routing, use
[Expert Parallelism](#expert-parallelism). The scheme-level meters
remain live for the selected model and hardware.

## Some Problems to Work

Let's use LLaMA-2 13B as a basic model for this section. Here are the model
details:✦adaptation Every
value in this table is scrubbable, and every answer below is computed from it
live: the exercises grade themselves against whatever model you dial in; each
question's stated givens (a batch size, a chip count) stay pinned, the way a
problem set's givens should. The preset button under the table restores the
chapter's LLaMA-2 13B.

| hyperparam | value |
| --- | --- |
| *L* |  |
| *D* |  |
| *F* |  |
| N |  |
| K |  |
| H |  |
| V |  |

Restore:
LLaMA-2 13B (chapter values)

LLaMA-2 has separate embedding and output matrices and a gated MLP block.

**Question 1:** How many parameters does LLaMA-2 13B have (I know
that's silly but do the math)? *Note that, as in
[Transformer Math](https://jax-ml.github.io/scaling-book/transformers/),
LLaMA-3 has 3 big FFW matrices, two up-projection and one down-projection. We
ignored the two "gating" einsum matrices in this section, but they behave the same
as Win in this section.*

Click here for the answer.

* FFW parameters: 3*L**D**F* =
* Attention parameters: 2*D*H*L*·(N + K) =
  Δedited The
  chapter reads: “Attention parameters: 4DNHL =
  4.2e9” — generalized to 2DHL·(N + K) so the count stays correct when
  you scrub the KV-head count K below the query-head count N (grouped-query
  attention). When K = N the two formulas are equal: 2DHL·(N + N) =
  4DNHL.
* Vocabulary parameters: 2V*D* =
* Total:
   +
   +
   =
  ,
  as expected!

**Question 2:** Let's assume we're training with BS=
tokens and using Adam. Ignoring parallelism for a moment, how much total memory is
used by the model's parameters, optimizer state, and activations? *Assume we
store the parameters in bf16 and the optimizer state in fp32 and checkpoint
activations three times per layer (after the three big matmuls).*

Click here for the answer.

The total memory used for the parameters (bf16) and the two optimizer states
(fp32, the first and second moment accumulators) is (2 + 4 + 4) ·
 ≈
.
The activations after the first two matmuls are shaped
BF and after the last one BD
(per the Transformer diagram above), so the total memory for bf16 is
2 · *L* · (*B**D* + 2 · *B**F*) =
2*L**B* · (*D* + 2*F*) or
2 ·  ·
 ·
 ·
(1 + 2 · ) ≈
 =
,
since B=. All other
activations are more or less negligible.✦adaptation Try:
drag the batch
and watch: the parameter + optimizer term
()
never moves, while the activation term scales linearly with it. That memory
monster is what FSDP-style activation sharding exists to slay.

**Question 3:** Assume we want to train with 32k sequence length
and a total batch size of 3M tokens on a TPUv5p 16x16x16 slice. Assume we want to
use bfloat16 weights and a float32 optimizer, as above.

1. Can we use pure data parallelism? Why or why not?
2. Can we use pure FSDP? Why or why not? With pure FSDP, how much memory will
   be used per device (assume we do gradient checkpointing only after the 3 big
   FFW matrices).
3. Can we use mixed FSDP + tensor parallelism? Why or why not? If so, what
   should *DP* and *TP* be? How much memory
   will be stored per device? Using only roofline FLOPs estimates and ignoring
   attention, how long will each training step take at
    MFU?

Click here for the answer.

First, let's write down some numbers. With 32k sequence length and a 3M batch
size, we have a sequence batch size of
.✦adaptation The
chapter says 96, which is 3·220/32,768; the live math here uses a
literal 3e6, which gives
. Either way:
small! Long contexts eat a token budget fast. On a TPU v5p
16x16x16 slice, we have
 of HBM.

1. We can't use pure data parallelism, because it replicates the parameters
   and optimizer states on each chip, which are already around
   (from Q2) which is more HBM than we have per-chip
   ().

   right now: 10 · params =
   per chip, vs
    of HBM →
   (max ~ params
   under pure DP + Adam; we have
   )
2. Let's start by looking purely at memory. Replacing
   BS= with 3M in
   Q2, we get
   ~
   total checkpoint activations, and with the
   optimizer state this brings us to almost exactly
    =
   .
   The TPUv5p slice has
    of HBM in
   total, so we are safely under the HBM limit.

   right now: training state
   sharded over 4096 chips =
   per chip, vs
    →

   Next let's look at whether we'll be comms or compute-bound. With 4096
   chips and 3 axes of parallelism, we can do a minimum batch size of
    · 4096 =
   tokens. That's slightly above our 3M batch size. So we're actually
   comms-bound, which is sad. So the general answer is **no, we cannot do
   FSDP alone**.

   right now: batch 3M vs floor
    →
3. Now we know our primary concern is being comms-bound, so let's plug in
   some numbers. First of all, we know from above that our per-chip batch size
   with mixed FSDP + tensor parallelism needs to be above
   ² / 2*F* =
   here. That means we can in theory do this! Let's figure out how much of
   each.

   right now: threshold
   tokens/chip, vs
   3M4096 =
    tokens/chip →

   We have the rule

   *DP*opt =
   *B**F* · *MDP**MTP* · *N*,

   so here we have
   sqrt(3e6 · 2 · 4096 / ) =
   ,
   meaning we'll do roughly
    way
   DP and
    way
   TP. Per TPU memory will be as in (2), and step time will just be

   6 · 3e6 · 4096 ·  ·  =
   ✦adaptation The
   numerator's "params" is Q1's live total,
   (the chapter's 13e9), so this line recomputes if you rewire the model in
   the table above. The MFU inside the fraction is scrubbable too.

**That's it for Part 5! For Part 6, which applies
this content to real LLaMA models,
[click here](https://jax-ml.github.io/scaling-book/applied-training/)!**✦adaptation Try:
open answer (3) above and drag
*F* =
down. A skinnier MLP raises the mixed-scheme threshold α²/2*F*
(tensor parallelism has less *F* to hide behind) *and*
shifts *DP*opt — watch the verdict pill flip when
the model gets too narrow for even the mixed scheme to save this small
batch.

## Appendix

### Appendix A: Deriving the backward pass comms

Above, we simplified the Transformer layer forward pass as
Out[B, D] = In[B, D] ·D Win[D, F] ·F Wout[F, D].
How do we derive the comms necessary for the backwards pass?

This follows fairly naturally from the rule in the previous section for a
single matmul Y = X · A:✦adaptation
In this appendix X and Y are the input and output *matrices* of a generic
matmul — the chapter's letters, kept as-is since this edition's mesh axes go by
*DP* and *TP*, so nothing collides.

dLdA =
dLdY ·
dYdA =
XT dLdY

dLdX =
dLdY ·
dYdX =
dLdY AT

Using this, we get the following formulas (letting
Tmp[B, F] stand for
In[B, D] · Win[D, F]):

1. dWout[F, D] = Tmp[B, F] ·B dOut[B, D]
2. dTmp[B, F] = dOut[B, D] ·D Wout[F, D]
3. dWin[D, F] = In[B, D] ·B dTmp[B, F]
4. dIn[B, D] = dTmp[B, F] ·F Win[D, F]

Note that these formulas are mathematical statements, with no mention of
sharding. The job of the backwards pass is to compute these four quantities. So
to figure out the comms necessary, we just take the shardings of all the
quantities which are to be matmulled in the four equations above (Tmp, dOut,
Wout, Win), which are specified by our parallelization
scheme, and use the rules of sharded matmuls to figure out what comms we have to
do. Note that dOut is sharded in the same way as Out.

Look back at
[Part 4: Transformer Math](https://jax-ml.github.io/scaling-book/transformers/),
continue to
[Part 6: Applied Training](https://jax-ml.github.io/scaling-book/applied-training/),
which works this content through real LLaMA models, or revisit the
[original chapter](https://jax-ml.github.io/scaling-book/training/) this
page adapts.

This page is an interactive edition of
[“How to Parallelize a Transformer for Training,”](https://jax-ml.github.io/scaling-book/training/)
Part 5 of *How to Scale Your Model* by Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya,
Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh, Albert Webson, and Reiner Pope
(Google DeepMind, 2025), with GPU material drawn from
[Chapter 12 of the same book](https://jax-ml.github.io/scaling-book/gpus/).
Source passages are the authors’, reproduced under the
[MIT License](LICENSE-scaling-book.txt) (© 2022 Maruan Al-Shedivat, © 2025 Google LLC),
whose notice ships with this page; departures are labeled as adaptation
(✦ additions, Δ edits). This AI-assisted interactive edition was initially
built by **Fable** (Claude, Anthropic) with its human editor and
adversarially reviewed and corrected by **OpenAI Codex**. Errors
introduced by the adaptation are not attributable to the book’s authors.

The interaction style follows Bret Victor’s
[Explorable Explanations](https://worrydream.com/ExplorableExplanations/):
a reactive document “allows the reader to play with the author’s assumptions and analyses, and see the consequences.”
No libraries, no build step, no network — view source and hack away.