# LLM Corpus Expansion

Working folder for the 506-URL reading-list expansion (started 2026-08-29).

- **[handoff.md](handoff.md)** — session handoff: current state, decisions and who made
  them, load-bearing code references, traps, rejected approaches, and next steps.
  Read this first.

## data/

Evidence captured from the session. These are inputs to the routing pass, not outputs.

| File | What it is |
|---|---|
| `source-urls-506.txt` | The original reading list, verbatim. 506 unique URLs. |
| `ingest-outcomes.tsv` | Per-URL result of the DEVONthink ingest: `MARKDOWN`, `BOOKMARK`, or `SKIP-DUP`. |
| `devonthink-inventory.tsv` | Current contents of the DEVONthink `Inference Optimization` group — record type, word count, URL, title. **The input the routing pass reads.** |

No routing has been performed yet. A keyword classifier was tried and rejected as
insufficiently accurate; its output was deliberately not committed. See the
"Rejected approaches" section of the handoff.
