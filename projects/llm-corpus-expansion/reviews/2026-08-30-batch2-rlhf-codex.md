# Independent review — LLM Fine-tuning batch 2 (preference and RL post-training)

- **Reviewer**: OpenAI Codex, codex-cli 0.149.0, report-only
- **Job**: task-mtf765dn-n2pf8c · session 01a05085-b590-7850-b8d0-f661f6c5ecdd
- **Verdict**: needs-rework — 3 HIGH, 4 MEDIUM, 1 LOW. All 8 applied.

## Brief

```markdown
You are the REVIEWER agent for a knowledge-base ingest. REPORT-ONLY: report findings, do
NOT edit any file. All paths are absolute; read them yourself.

## Schema you check against
/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/CLAUDE.md  (see "Independent agent review" for priorities and severities)

## The four Raw sources ingested (the ONLY permitted evidence)
/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Raw/blog-illustrating-reinforcement-learning-from-human-feedback-rlhf.md
/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Raw/doc-reinforcement-learning-rl-guide-unsloth-documentation.md
/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Raw/pdf-250301067-all-roads-lead-to-likelihood-the-value-of-reinforcement-learning-in-fi.md
/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Raw/doc-rlhf-learning-resources-in-2024-by-nathan-lambert.md

## Files created or edited
Created: /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/summaries/hf-illustrating-rlhf.md
         /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/summaries/unsloth-rl-grpo-guide.md
         /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/summaries/all-roads-lead-to-likelihood-2025.md
         /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/summaries/lambert-rlhf-resources-2024.md
         /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/concepts/Reward Model.md
Rewritten from stubs: /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/concepts/RLHF.md
                      /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/concepts/GRPO.md
                      /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/concepts/DPO.md
Edited:  /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/index.md, /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Wiki/log.md,
         /Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning/Learning Path/5a - Preference and RL Track.md,
         wiki_refs on the four Raw files

## Check priorities in order
1. FACTUAL ACCURACY / NO INVENTION — every claim, number and name must trace to one of
   the four sources. Outside knowledge counts as invention however true it is. The
   previous batch failed here by importing FSDP mechanics not present in its sources.
2. SCOPE DISCIPLINE  3. SCHEMA CONFORMANCE (type/tags/sources objects/updated/generated
   stamp/relative md links)  4. GRAPH CONNECTIVITY  5. CONSISTENCY (wiki_refs match real
   citers; log touched list accurate; SKELETON markers cleared where populated)
6. VOICE (concision, no emojis, no broken markdown)

## Attack these specifically
- Reward-model sizes (OpenAI 175B/6B, Anthropic 10B-52B, DeepMind 70B Chinchilla).
- The reward formula r = r_theta - lambda * r_KL and the claim that without the KL term
  the policy can produce gibberish that fools the reward model.
- Action space "~50k tokens" and the claim the initial model is untouched by gradients.
- The GRPO claim that BOTH the reward model and the value model are removed.
- FP8 RL (Nov 2025) and 380K-context gpt-oss (Jan 2026) dates.
- The "All Roads" thesis: that you can only LOSE information through a reward model, that
  on-policy sampling creates no new information, and the generation-verification-gap
  conclusion. Is the wiki's phrasing the paper's conclusion or my paraphrase?
- The DPO page explicitly says the DPO paper is not in this corpus. Verify nothing on
  that page states DPO mechanics unsupported by the "All Roads" paper.
- The cross-link from GRPO to the previous batch's LoRA rank-1 RL finding — is that
  claim, as restated, still faithful?

## Output
Severity-tagged findings (HIGH / MEDIUM / LOW) with file + line, then one verdict:
ship / ship-with-fixes / needs-rework.
```

## Findings as returned

## Findings

- **HIGH — Reward-model size relationship is misstated.** [hf-illustrating-rlhf.md:37](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/summaries/hf-illustrating-rlhf.md:37) and [Reward Model.md:33](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/concepts/Reward%20Model.md:33) say reward models are “often much smaller” than the policy. The source says their sizes **vary relative to the policy**: only OpenAI’s 6B RM is smaller than its 175B LM; Anthropic used 10B–52B for both, and DeepMind used 70B for both. The numbers are correct, but the conclusion drawn from them is not.

- **HIGH — The generation-verification hypothesis is presented as settled causation.** [RLHF.md:47](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/concepts/RLHF.md:47), [Reward Model.md:44](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/concepts/Reward%20Model.md:44), and [DPO.md:32](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/concepts/DPO.md:32) call it the paper’s “answer,” say it is “exactly why” two-stage training works, or state the explanation without qualification. The paper calls H6 a hypothesis applicable to problems with a generation-verification gap, for which it found the “most support” or “least evidence against.” The index also overstates it as “why the two-stage route wins” at [index.md:19](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/index.md:19) and [index.md:38](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/index.md:38). The dedicated summary’s wording at lines 36–49 is substantially faithful; “the value is not extra information—it is a smaller search” is a concise paraphrase, not a quotation, and is accurate only with the paper’s conditional/hypothesis framing retained.

- **HIGH — GRPO imports and broadens evidence outside the four permitted sources.** [GRPO.md:31](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/concepts/GRPO.md:31) cites the previous batch’s rank-1 result, whose Raw source is absent from this page’s provenance and outside this review’s permitted evidence. The narrow statement about policy-gradient RL is faithful to the previous summary, but line 32 broadens it to “RL’s capacity demands are low” generally and invents the conclusion that GRPO and PEFT efficiencies therefore compound. Neither inference appears in the four permitted sources.

- **MEDIUM — Required academic-paper page is missing.** The All Roads source is explicitly an academic preprint, but no `Wiki/papers/` page or directory exists; [index.md:26](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/index.md:26) still says “None yet.” The ingest workflow requires a paper page for academic papers.

- **MEDIUM — Learning Path claims four sources but links only two source summaries.** [5a - Preference and RL Track.md:9](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Learning%20Path/5a%20-%20Preference%20and%20RL%20Track.md:9) declares four sources, while its reading list at lines 24–31 omits `unsloth-rl-grpo-guide.md` and `lambert-rlhf-resources-2024.md`. This creates Learning Path drift under the workflow’s every-source rule.

- **MEDIUM — Learning Path page lacks its ingest trust stamp.** [5a - Preference and RL Track.md:1](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Learning%20Path/5a%20-%20Preference%20and%20RL%20Track.md:1) has `updated` but no `generated` object. Other populated Learning Path stages carry that stamp, and OKF requires it on ingest-touched pages.

- **MEDIUM — All eight generated timestamps postdate the files’ creation/modification.** For example, [hf-illustrating-rlhf.md:11](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/summaries/hf-illustrating-rlhf.md:11), [Reward Model.md:14](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/concepts/Reward%20Model.md:14), and [RLHF.md:17](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/concepts/RLHF.md:17) claim `2026-08-30T03:00:00Z`; all eight files were modified at `2026-08-30T02:34:40Z`, and the review began before 03:00Z. The same future stamp appears in the other three summaries and the GRPO/DPO concepts, so it is not a truthful generation time.

- **LOW — “No claims” wording contradicts the summary itself.** [lambert-rlhf-resources-2024.md:32](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/summaries/lambert-rlhf-resources-2024.md:32) says no claims are derived, although lines 20–30 derive three navigational/editorial claims. The same wording appears in [log.md:4](/Users/hamidadesokan/Documents/Knowledge%20Management/LLM%20Fine-tuning/Wiki/log.md:4). “No downstream technical claims” would match the apparent intent.

The attacked KL formula, gibberish failure, approximately 50K-token action space, untouched reference model, GRPO removal of both models, FP8/380K dates, and DPO mechanics are all supported by the permitted sources. Raw `wiki_refs`, local links, YAML, index presence, and SKELETON cleanup otherwise passed.

**Verdict: needs-rework**

Codex session ID: 01a05085-b590-7850-b8d0-f661f6c5ecdd
Resume in Codex: codex resume 01a05085-b590-7850-b8d0-f661f6c5ecdd
