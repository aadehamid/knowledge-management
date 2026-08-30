#!/usr/bin/env python3
"""Batch 1 ingest — LLM Fine-tuning: LoRA fundamentals and hyperparameters."""
import re
from pathlib import Path

V = Path("/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning")
TODAY = "2026-08-29"

RAW = {
    "tml":     "doc-lora-without-regret-thinking-machines-lab.md",
    "trl":     "doc-lora-without-regret-hugging-face.md",
    "unsloth": "doc-lora-fine-tuning-hyperparameters-guide-unsloth-documentation.md",
    "qdora":   "doc-efficient-finetuning-of-llama-3-with-fsdp-qdora-answerai.md",
}
TITLE = {
    "tml":     "LoRA Without Regret - Thinking Machines Lab",
    "trl":     "LoRA Without Regret (TRL guide) - Hugging Face",
    "unsloth": "LoRA fine-tuning Hyperparameters Guide - Unsloth Documentation",
    "qdora":   "Efficient finetuning of Llama 3 with FSDP QDoRA - Answer.AI",
}
SUMMARY = {
    "tml":     "lora-without-regret-tml-2025",
    "trl":     "lora-without-regret-trl-guide",
    "unsloth": "unsloth-lora-hyperparameters-guide",
    "qdora":   "fsdp-qdora-answerai-2024",
}


def src_block(keys, depth):
    """OKF provenance objects. depth = how many levels below the bundle root."""
    up = "../" * depth
    out = ["sources:"]
    for k in keys:
        out += [f"  - id: {RAW[k][:-3][:38]}",
                f"    resource: {up}Raw/{RAW[k]}",
                f'    title: "{TITLE[k]}"']
    return "\n".join(out)


def y(v: str) -> str:
    """Quote a YAML scalar when it needs it — same rule as sync_to_vault.yaml_scalar.
    An unquoted 'a: b' is a mapping, not a string, and blows up the parser."""
    if not re.search(r'[:#\[\]{}&*!|>%@`\'"]|^\s|\s$|^$', v):
        return v
    if '"' not in v:
        return '"' + v.replace("\\", "\\\\") + '"'
    return "'" + v.replace("'", "''") + "'"


def page(path: Path, fm_type, title, desc, tags, keys, body, depth):
    text = (f"---\ntype: {fm_type}\ntitle: {y(title)}\n"
            f"tags: [{', '.join(tags)}]\n{src_block(keys, depth)}\n"
            f"updated: {TODAY}\ndescription: {y(desc)}\n---\n\n{body.strip()}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


S = V / "Wiki" / "summaries"
E = V / "Wiki" / "entities"
C = V / "Wiki" / "concepts"

# ---------------------------------------------------------------- summaries
page(S / f"{SUMMARY['tml']}.md", "summary", "LoRA Without Regret (Thinking Machines Lab)",
     "LoRA matches full fine-tuning when applied to all layers and given ~10x the learning rate; capacity, not the method, is the limit",
     ["llm-finetuning", "lora", "peft", "reinforcement-learning"], ["tml"], """
# LoRA Without Regret (Thinking Machines Lab)

The central claim: LoRA is not inherently a performance compromise. Get a few details
right and it matches full fine-tuning (FullFT) in both sample efficiency and final loss,
across a "low-regret regime" the authors argue covers most post-training work.

LoRA replaces each weight matrix `W` with `W' = W + (alpha/r) * B A`, where `B` and `A`
are low-rank and `alpha` is a scaling constant.

## What the experiments found

- **Small-to-medium SFT: LoRA equals FullFT.** Measured on Tulu3 (instruction following)
  and OpenThoughts3 (reasoning), sweeping rank 1-512 and learning rate, using Llama 3 and
  Qwen3 models including an MoE.
- **Past its capacity, LoRA degrades gradually.** Exceeding the adapter's capacity does
  not hit a hard loss floor; it produces worse training efficiency, scaling with the
  relationship between capacity and dataset size.
- **LoRA tolerates large batches less well.** The gap versus FullFT widens with batch
  size and is *not* closed by raising the rank — the authors attribute it to the
  product-of-matrices (`BA`) parametrization having different optimization dynamics than
  optimizing `W` directly. Both methods do best at smaller batches, so this may not bite
  in practice.
- **Apply LoRA to all layers, especially MLP/MoE.** Attention-only LoRA underperforms
  even at matched parameter count: attention at rank 256 (0.25B params) loses to MLP-only
  at rank 128 (0.24B). This revises the original LoRA paper's attention-only advice.
- **For RL, even rank 1 suffices.** Policy-gradient RL matched FullFT at very low ranks.
  The argument is information-theoretic: supervised learning supplies O(tokens) bits per
  episode while policy gradient supplies O(1). Their MATH run — ~10,000 problems x 32
  samples — needs to absorb about 320,000 bits, while rank-1 LoRA on Llama-3.1-8B already
  carries ~3M parameters.

## Hyperparameters

- **Optimal LoRA learning rate is about 10x the FullFT rate.** Fitted across 14 Llama and
  Qwen models, the multiplier came out at 9.8. This makes transferring a known FullFT
  learning rate straightforward.
- **The `1/r` prefactor makes the optimal LR roughly rank-independent.** Early in
  training the learning curve is identical regardless of rank — close enough that the
  authors initially suspected a bug ignoring the rank parameter.
- **Four hyperparameters collapse to two.** `alpha`, `LR_A`, `LR_B` and `init_A` are
  linked by an invariance, leaving two effective degrees of freedom: `alpha * init_A *
  LR_B` (the initial update scale) and `init_A / LR_A` (how many steps it takes to move
  `A` off its initialization).
- They used `alpha = 32` with the standard `peft` initialization and **could not improve
  on it**.
- Because `B` starts at zero, the *effective* learning rate rises over training as `B`
  grows; by the end of their runs `B` had a larger spectral norm than `A`.

## Where this sits relative to other guidance

The article explicitly reinterprets two pieces of prior advice inside its two-parameter
basis: LoRA+ (a higher LR on `B`) and
[Unsloth's guidance](unsloth-lora-hyperparameters-guide.md) (higher `alpha` at high rank)
both amount to increasing `init_A / LR_A`. The recommendations differ in form, not in
kind — see the note in [LoRA](../entities/LoRA.md).

Reproduction recipes in TRL: [LoRA Without Regret (TRL guide)](lora-without-regret-trl-guide.md).
""", 2)

page(S / f"{SUMMARY['trl']}.md", "summary", "LoRA Without Regret (TRL guide)",
     "Hugging Face's TRL walkthrough reproducing the Thinking Machines LoRA results",
     ["llm-finetuning", "lora", "trl"], ["trl"], """
# LoRA Without Regret (TRL guide)

Hugging Face's practical companion to
[the Thinking Machines article](lora-without-regret-tml-2025.md), giving TRL
implementations of its findings so they can be reproduced rather than taken on trust.

- Frames the source result as: LoRA can match full fine-tuning **while using about 67% of
  the compute**, when configured correctly.
- Walks through SFT reproductions on Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct
  against `allenai/tulu-3-sft-mixture` and `open-thoughts/OpenThoughts-114k` — the same
  model and dataset pairings as the original.
- Recaps the mechanics: adapter layers over a frozen base, lower GPU memory, and the
  configuration care needed to avoid the performance trade-off that LoRA was once assumed
  to carry.

Useful mainly as the executable form of the research page; the reasoning and evidence
live in the [Thinking Machines summary](lora-without-regret-tml-2025.md).
""", 2)

page(S / f"{SUMMARY['unsloth']}.md", "summary", "Unsloth LoRA Hyperparameters Guide",
     "practitioner defaults for rank, alpha, learning rate, epochs and effective batch size",
     ["llm-finetuning", "lora", "qlora", "hyperparameters"], ["unsloth"], """
# Unsloth LoRA Hyperparameters Guide

Operational defaults rather than experimental evidence: what to set, with the stated goal
of raising accuracy while avoiding overfitting and underfitting.

## Recommended settings

| Parameter | Guidance |
|---|---|
| Learning rate | Typical range `2e-4` to `5e-6`. Start at `2e-4` for LoRA/QLoRA; `5e-6` for RL (DPO, GRPO); lower again for full fine-tuning. |
| Epochs | 1-3. Beyond 3, diminishing returns and rising overfitting risk on instruction data. |
| Rank `r` | 8 or 16 for fast runs, up to 128. Too large costs memory and speed and can overfit. |
| `lora_alpha` | Set equal to the rank, or double it. Keep `alpha/rank >= 1`. |
| rsLoRA | `use_rslora = True` switches scaling to `alpha/sqrt(r)`, which can help stability at higher ranks. |

The guide notes low learning rates are not merely slow — it argues they can themselves
lead to overfitting or prevent learning, rather than only causing underfitting.

## LoRA vs QLoRA

LoRA trains in 16-bit; QLoRA in 4-bit. QLoRA is slightly slower and marginally less
accurate but uses roughly **4x less VRAM** — the guide cites 70B Llama fitting in under
48GB. See [QLoRA](../entities/QLoRA.md).

## Effective batch size

`effective_batch_size = batch_size * gradient_accumulation_steps`. All factorizations of
the same product are equivalent for the weight update but not for memory, so the advice is
to keep `batch_size` small and raise `gradient_accumulation_steps` to avoid OOM. Unsloth
notes it shipped a fix making gradient accumulation and batch size genuinely equivalent,
which the guide says was not the case with standard gradient accumulation.

## Relation to the research

The alpha guidance here (alpha = rank or 2x rank) is stated as a heuristic. The
[Thinking Machines analysis](lora-without-regret-tml-2025.md) reads that advice as
equivalent to increasing `init_A / LR_A` in its two-parameter basis, and reports that it
could not beat a fixed `alpha = 32` with the standard `peft` parametrization. Both can
hold: they are different points in the same invariant space, not a factual disagreement.
""", 2)

page(S / f"{SUMMARY['qdora']}.md", "summary", "FSDP QDoRA (Answer.AI)",
     "quantized DoRA with FSDP: PEFT memory cost, claimed full-fine-tuning accuracy for continued pretraining",
     ["llm-finetuning", "qlora", "dora", "fsdp", "distributed-training"], ["qdora"], """
# FSDP QDoRA (Answer.AI)

Answer.AI's follow-up to FSDP/QLoRA, which had made 70B fine-tuning possible on gaming
GPUs. QDoRA is quantized DoRA (Weight-Decomposed Low-Rank Adaptation) with FSDP support.

- The claim: QDoRA is **as memory-efficient and scalable as FSDP/QLoRA while being as
  accurate as full-weight training for continued pretraining** — positioned as closing the
  gap between parameter-efficient and full fine-tuning.
- On mathematical data, Llama-3-8B with QDoRA or full fine-tuning is reported to greatly
  outperform QLoRA and Llama 2 — with the caveat, stated in the source, that full
  fine-tuning uses far more memory.
- Earlier Llama-2 7B results on Orca-Math showed QDoRA ahead of the other methods at much
  lower memory than full fine-tuning. These are described as preliminary and **without
  hyperparameter tuning**.
- The same release adds quantized Llama-Pro (Progressive LLaMA with Block Expansion) with
  FSDP support.

Framed by the authors as combining much of QLoRA's parameter efficiency with the more
granular optimization of full fine-tuning. Treat the accuracy claim as the authors'
preliminary result rather than a settled comparison — the source itself calls the
experiments early.

See [FSDP](../entities/FSDP.md) and [QLoRA](../entities/QLoRA.md).
""", 2)

# ---------------------------------------------------------------- entities
page(E / "LoRA.md", "entity", "LoRA",
     "low-rank adaptation: train two small matrices instead of full weights",
     ["llm-finetuning", "lora", "peft"], ["tml", "unsloth", "trl"], """
# LoRA

Low-Rank Adaptation freezes the pretrained weights and learns a low-rank update instead.
Each weight matrix `W` becomes `W' = W + (alpha/r) * B A`, where `A` and `B` are rank-`r`
matrices holding far fewer parameters than `W`.

## Why it is used

- **Multi-tenant serving.** The base weights are untouched, so one inference server can
  hold many adapters at once and sample from them in a batch. vLLM and SGLang implement
  this.
- **Smaller training layout.** Full fine-tuning must store optimizer state and gradients
  for every weight, often in float32, typically needing an order of magnitude more
  accelerators than serving the same model. LoRA trains on a layout only slightly larger
  than inference.
- **Cheap to move.** Adapters are small, so they transfer and load quickly.

## How to configure it

- **Apply it to all weight matrices, especially MLP and MoE layers.** Attention-only LoRA
  underperforms even when parameter count is matched. This supersedes the original 2021
  paper's attention-only recommendation.
- **Use roughly 10x the full-fine-tuning learning rate.** Fitted at 9.8x across 14 Llama
  and Qwen models.
- **Rank matters less than expected for the learning rate.** The `1/r` prefactor makes the
  optimal LR approximately rank-independent; early learning curves are near-identical
  across ranks. Rank governs *capacity*, and therefore how large a dataset the adapter can
  absorb before efficiency degrades.
- Practitioner starting points: rank 8-16 for quick runs, up to 128; `alpha` equal to or
  double the rank; learning rate `2e-4`.

## A difference in recommendations, not in fact

[Unsloth](../summaries/unsloth-lora-hyperparameters-guide.md) recommends raising `alpha`
with rank; [Thinking Machines](../summaries/lora-without-regret-tml-2025.md) fixed
`alpha = 32` and could not improve on it. The latter shows these are the same point in a
two-parameter invariant space — raising `alpha` is equivalent to increasing
`init_A / LR_A`. Neither is wrong; they differ in which knob is held fixed.

## Where LoRA falls short

Large batch sizes: the gap to full fine-tuning widens with batch size and raising the rank
does not close it, because it stems from the `BA` parametrization's optimization dynamics.

For reinforcement learning, capacity demands collapse — rank 1 matched full fine-tuning,
consistent with policy gradients carrying O(1) bits of information per episode.

Related: [QLoRA](QLoRA.md), [PEFT](PEFT.md),
[Supervised Fine-Tuning](../concepts/Supervised%20Fine-Tuning.md).
""", 2)

page(E / "QLoRA.md", "entity", "QLoRA",
     "LoRA over a 4-bit quantized base model, trading a little accuracy for ~4x less VRAM",
     ["llm-finetuning", "qlora", "quantization", "peft"], ["unsloth", "qdora"], """
# QLoRA

LoRA applied on top of a quantized base model. Where [LoRA](LoRA.md) trains in 16-bit,
QLoRA holds the frozen base in 4-bit.

- **Roughly 4x less VRAM** than 16-bit LoRA, at the cost of being slightly slower and
  marginally less accurate. Unsloth cites a 70B Llama fitting in under 48GB.
- The practical rule from the same guide: prefer LoRA when you have the memory and want
  maximum accuracy; prefer QLoRA when memory is the binding constraint.

## FSDP and the descendants

Answer.AI's FSDP/QLoRA first made 70B fine-tuning feasible on gaming GPUs by combining
quantization with sharded training. Its successor **QDoRA** — quantized
Weight-Decomposed Low-Rank Adaptation — is claimed to keep that memory profile while
matching full-weight training accuracy for continued pretraining, which would remove the
main reason to prefer full fine-tuning. Those results are preliminary and untuned by the
authors' own description; see [FSDP QDoRA](../summaries/fsdp-qdora-answerai-2024.md).

Related: [FSDP](FSDP.md), [PEFT](PEFT.md).
""", 2)

page(E / "PEFT.md", "entity", "PEFT",
     "parameter-efficient fine-tuning: adjust a small parameter set instead of all weights",
     ["llm-finetuning", "peft", "lora"], ["tml", "unsloth"], """
# PEFT

Parameter-Efficient Fine-Tuning: the family of methods that adapt a large network by
updating a much smaller set of parameters, and the Hugging Face library that implements
them.

The motivating intuition is an accounting one — post-training uses far smaller datasets
over narrower domains than pretraining, so spending a terabit of weights to represent
updates learned from a gigabit or megabit of data looks wasteful.

[LoRA](LoRA.md) is the leading method in this family. [QLoRA](QLoRA.md) adds
quantization of the frozen base; DoRA decomposes the weight update further.

The `peft` library's standard LoRA parametrization — uniform initialization of `A` scaled
by `1/sqrt(d_in)`, `B` initialized to zero, one learning rate for both, `alpha = 32` — is
what the [Thinking Machines experiments](../summaries/lora-without-regret-tml-2025.md)
used, and they were unable to improve on it.
""", 2)

page(E / "FSDP.md", "entity", "FSDP",
     "PyTorch Fully Sharded Data Parallel: shards parameters, gradients and optimizer state across GPUs",
     ["llm-finetuning", "fsdp", "distributed-training"], ["qdora"], """
# FSDP

Fully Sharded Data Parallel, PyTorch's approach to splitting parameters, gradients and
optimizer state across GPUs so a model larger than one device's memory can be trained.

In this corpus FSDP appears mainly in combination with quantized PEFT:

- **FSDP/QLoRA** (Answer.AI) made 70B fine-tuning possible on gaming GPUs for the first
  time, by sharding a quantized base model.
- **FSDP QDoRA** extends the same machinery to quantized DoRA, and to quantized Llama-Pro
  block expansion. See [FSDP QDoRA](../summaries/fsdp-qdora-answerai-2024.md).

The reason this pairing matters is the memory asymmetry noted under [LoRA](LoRA.md): full
fine-tuning must hold optimizer moments and gradients for every weight, often in float32,
which is what pushes it an order of magnitude beyond the serving layout.

Related: [QLoRA](QLoRA.md).
""", 2)

print("summaries + entities written")
