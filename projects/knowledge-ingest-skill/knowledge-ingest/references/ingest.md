# The ingest loop in detail

## Choosing a batch

Find sources nothing cites yet:

```python
from pathlib import Path
B = Path("<bundle>")
cited = "".join(p.read_text(encoding="utf-8", errors="ignore")
                for p in (B / "Wiki").rglob("*.md"))
for r in sorted((B / "Raw").glob("*.md")):
    if r.name not in cited:
        print(len(r.read_text(encoding="utf-8", errors="ignore").split()), r.name)
```

Take 8–12 on one topic. The word counts matter: a batch of 200-word shells produces a
batch of thin pages, and it is better to know that before writing than after.

## Page types

| Type | Where | When |
|---|---|---|
| summary | `Wiki/summaries/<slug>.md` | one per source, or one per group of closely-related sources |
| concept | `Wiki/concepts/<Name>.md` | a mechanism or idea |
| entity | `Wiki/entities/<Name>.md` | a named thing — a library, model, tool |
| paper | `Wiki/papers/<Title>.md` | the source is an academic paper |

Folding several thin sources into one summary is usually right — three tool READMEs that
each say little make one good page and three bad ones. Say in the page that it covers
several sources, and cite them all.

Fill existing stub pages before creating new ones.

## Frontmatter

```yaml
---
type: summary
title: Some Title
tags: [topic, subtopic]
sources:
  - id: <source-file-stem, truncated>
    resource: ../../Raw/<source-file>.md
    title: "Source title"
updated: 2026-08-30
description: one sentence, lower case, no trailing period
generated: { by: "agent:<model-id>", at: "2026-08-30T14:22:05Z" }
---
```

Three things that break files if ignored:

- **Quote any value containing `:`**. An unquoted colon makes YAML read the value as a
  mapping and the file fails to parse.
- **`resource:` is relative to the page's own directory.** Two levels down needs `../../`.
  Wrong depth means broken provenance that no link checker in the body will catch.
- **`at:` is the real clock**, from `date -u +%Y-%m-%dT%H:%M:%SZ`. A rounded guess that
  lands after the file's own mtime is a stamp that lies, which is worse than none.

Links between pages are relative markdown with percent-encoded spaces:
`[Reward Model](../concepts/Reward%20Model.md)`.

## Wiring

Four things, all required, none of them optional-feeling in a way that survives review:

**Index** — every new page listed under its section. A page missing from the index is an
orphan.

**Back-references** — each source's frontmatter lists the pages citing it. This must hold
in *both* directions; the checker verifies both.

**Curriculum stage** — place each source in a stage. When a stage goes from empty to
populated, remove its skeleton marker, add a line naming how many sources it now holds,
and bump `updated`.

**Log** — one entry, under today's date heading, newest first, one heading per date. Say
what was ingested, what was created, and any judgement call you made. The log is what a
future agent reads to understand why the bundle looks the way it does.

## Writing

Read the source, then write from what you read. The gap where invention creeps in is the
sentence you write while thinking about the topic rather than looking at the text.

Signals you are drifting:
- You are explaining background the source assumed rather than stated.
- You reached for a number you remember rather than one you just read.
- You are smoothing two sources into one story.

All three produce pages that read better than the honest version and are worth less.
