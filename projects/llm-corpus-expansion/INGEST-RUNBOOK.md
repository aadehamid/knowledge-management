# Ingest runbook — this repo

**The method is the `knowledge-ingest` skill.** Read it first:
`projects/knowledge-ingest-skill/knowledge-ingest/SKILL.md`, plus the `references/` file
for whatever mode you are in (ingest, route, bootstrap, audit).

This file holds only what is specific to *this* repo — paths, conventions, commands and
the local decisions already made. Where the two ever disagree, the skill is the method and
this file is the local detail; nothing here restates the skill, on purpose, because two
copies of the same instruction drift and then an agent follows the wrong one.

If you read nothing else here, read **Local conventions** and **Commands**.

---

## Where things are

| What | Where |
|---|---|
| Repo | `/Users/hamidadesokan/Dropbox/1_PROJECTS/knowledge-management` |
| Vaults | `/Users/hamidadesokan/Documents/Knowledge Management/<Vault Name>/` |
| Python | `<repo>/.venv/bin/python` — has the conversion deps |
| Config | `<repo>/knowledge-ingest.config.json` — the skill reads this |
| Ledger | `projects/llm-corpus-expansion/INGEST-LEDGER.md` |
| Past reviews | `projects/llm-corpus-expansion/reviews/` |

Eight bundles: `Transformer from Scratch`, `LLM Fine-tuning`, `LLM Inference Optimization`,
`ML Foundations`, `AI Engineering`, `Document AI and Retrieval`, `CUDA from Scratch`,
`LLM Landscape` (catalog-shaped — no concept layer, by design).

Each has `Raw/` (sources; read-only except frontmatter `wiki_refs`), `Wiki/`,
`Learning Path/`, `CLAUDE.md` (that bundle's schema — authoritative for its own layout),
`index.md`.

**The vaults are NOT in git.** Deletions are unrecoverable. You should never need one; if
you think you do, that is a stop-and-ask.

---

## Local conventions

These are this repo's choices, not the skill's. The bundles follow **OKF v0.2**:

- Wiki pages carry `type` (`entity` | `concept` | `paper` | `summary` | `overview` |
  `index`), `title`, `description`, `tags`, `sources` (objects with `id`, `resource`,
  `title`), `updated`, and a `generated` trust stamp.
- Raw sources carry `type: Source` and `fetch_status` — **not** `status`, which OKF
  reserves for draft/stable/deprecated.
- `index.md` and `log.md` are **frontmatter-free** reserved files. Bundle-root `index.md`
  may carry only `okf_version`.
- `log.md` entries are date-grouped **newest first**, one heading per date. Add to today's
  heading if it exists rather than creating a second.
- Links are relative markdown with `%20` for spaces — **never `[[wikilinks]]`** outside
  `Raw/`. The validator rejects them.
- `resource:` paths are relative to the page's own directory: `../../Raw/…` from
  `Wiki/summaries/` or `Wiki/concepts/`, `../Raw/…` from `Wiki/overview.md`.

Full page templates: the skill's `references/ingest.md`.

---

## Commands

**Find unprocessed sources in a bundle**

```bash
python3 - <<'PY'
from pathlib import Path
V = Path("/Users/hamidadesokan/Documents/Knowledge Management/LLM Fine-tuning")
wiki = "".join(p.read_text(encoding="utf-8", errors="ignore")
               for p in (V / "Wiki").rglob("*.md"))
for r in sorted((V / "Raw").glob("*.md")):
    if r.name not in wiki:
        print(len(r.read_text(encoding="utf-8", errors="ignore").split()), r.name)
PY
```

**Gate A**

```bash
cd /Users/hamidadesokan/Dropbox/1_PROJECTS/knowledge-management
python3 scripts/check_ingest.py "<vault path>" \
  --validator scripts/test_okf_bundle.py --since $(date +%Y-%m-%d)
```

Must exit clean. Read the warnings — a *skipped* check is not a pass.

**Gate B — Codex is the reviewer of record here**

```bash
C=~/.claude/plugins/cache/openai-codex/codex/1.0.6/scripts/codex-companion.mjs
node "$C" task --background --effort high "$(cat /tmp/review_brief.md)"
node "$C" status <job-id> --json
node "$C" result <job-id>
```

It runs `write: false`, which satisfies report-only structurally rather than by
instruction. Archive brief and findings to `projects/llm-corpus-expansion/reviews/`
as `<date>-batch<N>-<topic>-codex.md`, and name the reviewer in the bundle's `log.md`.

**Convert and sync new sources** (after adding `urls.txt` lines)

```bash
.venv/bin/python scripts/convert_pdfs.py
NLM_SKIP=1 .venv/bin/python scripts/sync_to_vault.py
```

`markitdown[youtube-transcription]` is load-bearing — without the extra a 13,000-word
lecture silently converts to a 200-word shell. If video conversions come back at a few
hundred words, that extra is missing.

**Commit** — the vaults are not tracked, so this commits scripts, ledger and review
records, not the wiki pages themselves. That is expected.

---

## Local decisions already made

Do not relitigate these without reason:

- **`LLM Landscape` is catalog-shaped.** No concept layer. Its sources are dated releases
  and industry commentary; its top terms are borrowed from other subjects.
- **`Document AI and Retrieval`** was renamed from `rag-retrieval` — OCR appears 595 times
  against 138 for the next term.
- **NotebookLM is gated.** The five newer subjects carry `null` notebook ids, which blocks
  both auto-provisioning and push. Lift deliberately, per subject. Note that push only
  fires for files newly written on a sync, so a skipped push does not self-heal.
- **Review cadence** is currently every third batch, at the user's direction, with the
  exposure recorded in the ledger.

---

## Known open items

- Three Microway CUDA sources never converted (fetch failure).
- Five CUDA YouTube sources converted to 14-word shells and were ingested at that quality.
- NotebookLM is ~491 files behind; three existing notebooks need a deliberate backfill.
- Four bundles (`ML Foundations`, `AI Engineering`, `Document AI and Retrieval`,
  `LLM Landscape`) have **no reviewer named in their logs** — their fidelity has never been
  checked by anything but the agent that wrote them.

---

## Worked examples

- `projects/llm-corpus-expansion/ingest_batch3.py` — a complete batch: 8 sources, 5
  summaries, 2 paper pages, one stub filled. Copy its shape.
- `Wiki/concepts/Synthetic Data.md` in `LLM Fine-tuning` — the writing standard. Note how
  the ReST-EM result carries its enabling condition inline rather than being generalised,
  and how a dispute is recorded unresolved.
- `projects/llm-corpus-expansion/reviews/` — both Codex reviews verbatim, 16 findings
  between them. Read one before writing your first review brief.

## Background

- `INGEST-LEDGER.md` — what remains, batch order, standing decisions
- `handoff.md` — how the corpus reached this state
- `RECALL-SYNC.md` — diffing the Recall knowledge base against this corpus
- Each bundle's `CLAUDE.md` — authoritative for that bundle's own layout
