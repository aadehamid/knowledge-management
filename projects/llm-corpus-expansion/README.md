# LLM Corpus Expansion

Working folder for the 506-URL reading-list expansion (started 2026-08-29).

- **[AGENT-PROMPT.md](AGENT-PROMPT.md)** — the prompt to paste when starting a new agent
  on this work.
- **[INGEST-RUNBOOK.md](INGEST-RUNBOOK.md)** — **start here if you are the agent doing
  the ingest.** Step-by-step operating instructions, templates, the verification command,
  and the failure modes that past reviews actually caught.
- **[INGEST-LEDGER.md](INGEST-LEDGER.md)** — what is done, what remains, batch order.
  Update it at the end of every batch.
- **[routing.md](routing.md)** — the routing pass: all 546 records assigned to subject
  areas, with the eight open decisions that need your call.
- **[handoff.md](handoff.md)** — session handoff: current state, decisions and who made
  them, load-bearing code references, traps, rejected approaches, and next steps.
  Read this first.

## data/

Evidence captured from the session. These are inputs to the routing pass, not outputs.

| File | What it is |
|---|---|
| `source-urls-506.txt` | The original reading list, verbatim. 506 unique URLs. |
| `ingest-outcomes.tsv` | Per-URL result of the DEVONthink ingest: `MARKDOWN`, `BOOKMARK`, or `SKIP-DUP`. |
| `devonthink-inventory.tsv` | Contents of the DEVONthink `Inference Optimization` group — record type, word count, URL, title. The input the routing pass read. |
| `routing.tsv` | **The routing result.** Per record: subject, confidence, duplicate flag, whether the URL is already in a repo `urls.txt`. |
| `assignments-raw.tsv` | Bare `idx / subject-code / confidence` triples as assigned, before the join. |

Routing is complete (see `routing.md`); seeding has not started. A keyword classifier
was tried first and rejected as insufficiently accurate — see the "Rejected approaches"
section of the handoff.
