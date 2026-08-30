# Independent review — LLM Fine-tuning batch 1 (LoRA fundamentals)

- **Reviewer**: OpenAI Codex, codex-cli 0.149.0, report-only (`write: false`)
- **Author**: Claude Opus 5
- **Job**: task-mtf6f8m1-xwql8v · Codex session 01a05072-90d8-7341-a255-617c9c8ce022
- **Verdict**: needs-rework — 6 HIGH, 2 MEDIUM, 0 LOW. All 8 applied.

The vault is not git-tracked, so this file is the durable record of the review.
The applied fixes live in the LLM Fine-tuning vault; `Wiki/log.md` carries the
attribution required by the schema.

## Brief given to the reviewer

```markdown
You are the REVIEWER agent for a knowledge-base ingest. The review is REPORT-ONLY:
report findings, do NOT edit any file.

All paths below are absolute. Read them yourself.

## The schema you check against
/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/CLAUDE.md
(See its "Independent agent review" section for the check priorities and severity scale.)

## The four Raw sources that were ingested (the ONLY permitted evidence)
/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Raw/doc-lora-without-regret-thinking-machines-lab.md
/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Raw/doc-lora-without-regret-hugging-face.md
/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Raw/doc-lora-fine-tuning-hyperparameters-guide-unsloth-documentation.md
/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Raw/doc-efficient-finetuning-of-llama-3-with-fsdp-qdora-answerai.md

## Files created or edited in this ingest (what you are reviewing)
Created:
  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/summaries/lora-without-regret-tml-2025.md
  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/summaries/lora-without-regret-trl-guide.md
  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/summaries/unsloth-lora-hyperparameters-guide.md
  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/summaries/fsdp-qdora-answerai-2024.md
Rewritten from bootstrap stubs:
  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/entities/LoRA.md
  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/entities/QLoRA.md
  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/entities/PEFT.md
  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/entities/FSDP.md
Edited:
  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/index.md
  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/log.md
  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Learning Path/3 - Parameter-Efficient Methods.md
  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Learning Path/5b - Scaling and Systems Track.md
  wiki_refs frontmatter on the four Raw files listed above

## Check priorities, in order
1. FACTUAL ACCURACY / NO INVENTION — every claim, number, model name, dataset name and
   figure must be traceable to one of the four Raw sources. Flag anything plausible but
   unsourced. Be specific: quote the wiki claim and say what the source actually says.
2. SCOPE DISCIPLINE — out-of-scope material should be an aside, not a new page.
3. SCHEMA CONFORMANCE — frontmatter type/tags/sources/updated; sources[] as OKF objects
   with id, resource, title; relative markdown links (NOT [[wikilinks]]); absolute dates.
4. GRAPH CONNECTIVITY — no orphans; index entries present and sensible.
5. CONSISTENCY — each Raw file's wiki_refs must match the pages that actually cite it;
   the log "touched" list must match reality; Learning Path SKELETON markers correct.
6. VOICE — concision, no emojis, no broken markdown.

## Specific things to attack
- The claim "optimal LoRA learning rate is about 10x FullFT, fitted at 9.8x across 14
  Llama and Qwen models" — is that what the source says?
- The attention-vs-MLP parameter comparison (attn rank 256 = 0.25B vs MLP rank 128 =
  0.24B) — check the numbers against the source table.
- The RL information-theory argument (320,000 bits; rank-1 LoRA ~3M params) — check.
- "about 67% of the compute" attributed to the TRL guide — check attribution.
- The QDoRA accuracy claim — the wiki says it is preliminary and untuned. Verify the
  source supports that hedge, and that no stronger claim leaked in.
- The alpha disagreement between Unsloth and Thinking Machines — the wiki claims these
  are the same point in a two-parameter invariant space. Is that supported, or is it my
  own synthesis presented as sourced?

## Output format
Severity-tagged findings: HIGH (factual error / invented content / dangling link),
MEDIUM (schema or consistency violation), LOW (polish). Each with file + line.
End with one verdict: ship / ship-with-fixes / needs-rework.
```

## Findings as returned

## Findings

- HIGH #1 - Alpha guidance is over-reconciled. [LoRA.md:54](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/entities/LoRA.md:54), [Unsloth summary:47](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/summaries/unsloth-lora-hyperparameters-guide.md:47), and [log.md:4](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/log.md:4) claim the recommendations are the "same point," that neither is wrong, and that no disagreement exists. The source only maps higher alpha into the same two-dimensional framework: preserving update size requires reducing both learning rates, which increases `init_A / LR_A` ([Raw:165](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Raw/doc-lora-without-regret-thinking-machines-lab.md:165), [Raw:174](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Raw/doc-lora-without-regret-thinking-machines-lab.md:174)). That is a different point in the effective parameter space. The source does not conclude that both recipes are equivalent or equally correct.

- HIGH #2 - "Capacity, not the method, is the limit" is false as an absolute summary. [TML summary:10](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/summaries/lora-without-regret-tml-2025.md:10) and [index.md:31](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/index.md:31) omit the method-specific large-batch limitation. The source says this penalty is independent of rank and appears to arise from LoRA's product-of-matrices parametrization ([Raw:49](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Raw/doc-lora-without-regret-thinking-machines-lab.md:49), [Raw:82](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Raw/doc-lora-without-regret-thinking-machines-lab.md:82)).

- HIGH #3 - FSDP mechanics are invented relative to the permitted evidence. [FSDP.md:10](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/entities/FSDP.md:10), [FSDP.md:15](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/entities/FSDP.md:15), and [Stage 5b:23](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Learning%20Path/5b%20-%20Scaling%20and%20Systems%20Track.md:23) say FSDP shards parameters, gradients, and optimizer state so models larger than one device can train. None of the four Raw sources states this. Answer.AI explicitly sends readers to another article for FSDP fundamentals and only exposes a `full_shard` option ([Raw:43](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Raw/doc-efficient-finetuning-of-llama-3-with-fsdp-qdora-answerai.md:43), [Raw:158](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Raw/doc-efficient-finetuning-of-llama-3-with-fsdp-qdora-answerai.md:158)).

- HIGH #4 - FullFT learning-rate ordering is unsupported. [Unsloth summary:22](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/summaries/unsloth-lora-hyperparameters-guide.md:22) says FullFT should use a rate "lower again" after listing `5e-6` for RL. The source only says lower rates are generally more appropriate for FullFT; it does not place them below `5e-6` ([Raw:38](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Raw/doc-lora-fine-tuning-hyperparameters-guide-unsloth-documentation.md:38)).

- HIGH #5 - A tentative batch-size explanation is presented as settled causation. [LoRA.md:60](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/entities/LoRA.md:60) says the gap occurs "because" of `BA` optimization dynamics. The source calls this the "likely reason" and limits it to the tested dataset ([Raw:88](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Raw/doc-lora-without-regret-thinking-machines-lab.md:88)).

- HIGH #6 - The RL information estimate loses essential assumptions. [TML summary:39](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/summaries/lora-without-regret-tml-2025.md:39) presents 320,000 bits as what the run "needs to absorb." The arithmetic and approximately 3M rank-1 parameters are correct, but the source assumes one bit per completion, limits the claim to policy-gradient methods, and later calls the formal information estimate an upper bound ([Raw:121](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Raw/doc-lora-without-regret-thinking-machines-lab.md:121), [Raw:232](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Raw/doc-lora-without-regret-thinking-machines-lab.md:232), [Raw:242](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Raw/doc-lora-without-regret-thinking-machines-lab.md:242)).

- MEDIUM #7 - Required ingest trust stamps are absent. The four summaries, four rewritten entities, and both Learning Path pages close their frontmatter without `generated: { by, at }`; examples are [TML summary:11](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/summaries/lora-without-regret-tml-2025.md:11), [LoRA.md:17](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/entities/LoRA.md:17), and [Stage 3:5](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Learning%20Path/3%20-%20Parameter-Efficient%20Methods.md:5). [CLAUDE.md:409](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/CLAUDE.md:409) requires this stamp for every new ingest's touched pages.

- MEDIUM #8 - Both populated Learning Path stages retain stale `SKELETON` markers. See [Stage 3:9](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Learning%20Path/3%20-%20Parameter-Efficient%20Methods.md:9) and [Stage 5b:9](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Learning%20Path/5b%20-%20Scaling%20and%20Systems%20Track.md:9). [CLAUDE.md:172](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/CLAUDE.md:172) requires removing the marker and adding a `Populated with N sources` line.

No LOW findings.

The 9.8x/14-model fit, 0.25B-versus-0.24B comparison, TRL's approximately 67% compute statement, and preliminary/untuned QDoRA hedge are supported. No dangling links or orphans were found; index coverage, Raw `wiki_refs`, the log touched counts, and Markdown-link style are consistent.

Verdict: needs-rework

Codex session ID: 01a05072-90d8-7341-a255-617c9c8ce022
Resume in Codex: codex resume 01a05072-90d8-7341-a255-617c9c8ce022
