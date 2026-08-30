# Self-review: batches 4-6 traceability verification

Reviewer: Hamid + Hermes (self-review, all 15 claims verified against sources manually)
Date: 2026-08-30
Scope: All claims in unsloth-hardware-scaling.md, axolotl-wing-lian-interview.md, fsdp-qlora-answerai-2024-full.md, distributed-training-guide.md verified against their source documents.

## Findings

| # | Claim | Verdict | Action |
|---|---|---|---|
| 1 | Phi-4-mini ~5.3 → ~7.3 | CORRECT | none |
| 2 | Mistral Small baseline ~13 | CORRECT | none |
| 3 | Qwen 1.7B ~5 | CORRECT | none |
| 4 | Gemma 3 4B ~9 | CORRECT | none |
| 5 | "I don't recommend 4-bit... merging back adapters" | CORRECT (verbatim) | none |
| 6 | "maybe 75% confidence" | CORRECT (verbatim) | none |
| 7 | "Unsloth is single-GPU only" | CORRECT | none |
| 8 | Alpaca rerun: 8 A100s/3hr/$100 → 8 L40s/30min/$4-5 | CORRECT | none |
| 9 | "20x improvement sometimes" | CORRECT | none |
| 10 | "five or six tricks" | CORRECT (verbatim) | none |
| 11 | "90% of your time" | PARTIALLY CORRECT | fixed: "80 or 90% plus" |
| 12a-d | Model characterizations (Mistral/Gemma/Qwen/Llama 4) | CORRECT | none |
| 13 | Embeddings 24% / QKV fused | CORRECT | none |
| 14 | torch-compile conflict | CORRECT | none |
| 15 | Unsloth config won't load in vLLM | CORRECT | none |

Additional verified from Wing Lian interview: ReLoRA "didn't really have the impact",
QLoRA "not quite as good", Mamba "three days from paper drop" — all accurate paraphrases.

## Verdict: ship-with-fixes
1 finding applied (claim 11: 90% → 80 or 90%). No HIGH or MEDIUM findings.
The mechanical checker (check_ingest.py) is clean. The only remaining risk is
factual accuracy on the pages NOT covered by this spot-check (unsloth-gpt-oss-guide,
unsloth-studio, unsloth-vlm-rl, unsloth-qwen3-vl-guide, unsloth-gemma-family,
unsloth-llama3-ollama-tutorial, ms-swift-framework, axolotl-repo-and-configs,
philschmid-fsdp-qlora-llama3, consumer-hardware-finetuning, mlx-apple-silicon,
lora-concepts-video) — these were written from shorter, more structured sources
(official documentation) where the invention risk is lower than for the video
transcript.
