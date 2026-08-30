# Prompt for the ingest agent

Paste the block below to start a new agent on the ingest.

**When the `knowledge-ingest` skill is installed you should not need this.** Describing the
work is enough — the skill triggers on intent, reads `knowledge-ingest.config.json`, and
that config points it at the runbook. This prompt is the fallback for agents without the
skill: a different harness, a machine where it is not installed, or a session where it
fails to trigger.

It is deliberately short — the detail lives in `INGEST-RUNBOOK.md`, which the agent is told
to read first. What is repeated inline is only what must not be missed even if the runbook
is skimmed.

---

```text
You are continuing a knowledge-base ingest that is already underway. Work through it
batch by batch until the backlog is done.

FIRST ACTION — before anything else, read these two files completely:
  /Users/hamidadesokan/Dropbox/1_PROJECTS/knowledge-management/projects/llm-corpus-expansion/INGEST-RUNBOOK.md
  /Users/hamidadesokan/Dropbox/1_PROJECTS/knowledge-management/projects/llm-corpus-expansion/INGEST-LEDGER.md
The runbook is authoritative. Where your judgement and the runbook disagree, follow the
runbook.

THE JOB
Eight Obsidian vaults live at "/Users/hamidadesokan/Documents/Knowledge Management/".
Each has Raw/ (fetched sources), Wiki/ (what you write) and Learning Path/ (curriculum).
Turn unprocessed Raw sources into interlinked wiki pages, 8-12 related sources per batch.
About 518 sources remain. The ledger says which cluster is next.

THE RULE THAT MATTERS MOST
Every factual sentence you write must be traceable to a line in one of THAT BATCH's
source files. Not "true", not "well known" — in the source, in this batch.
- Keep the source's hedges. "likely", "suggests", "on this dataset", "assuming X",
  "we found the most support for" are part of the claim, not decoration.
- Check every number and every comparative word ("smaller", "faster", "most") against
  the source before writing it.
- If you would write a sentence without the source open, delete it.
- If two sources disagree, record both and say they disagree. Never pick a winner.
- If no source covers something, write that the sources do not cover it. That is a
  legitimate and useful sentence.
Two independent reviews of earlier batches both failed on exactly this. The runbook
shows the five specific sentences that were rejected and what the sources actually said.
Read that table.

AFTER EVERY BATCH
  cd /Users/hamidadesokan/Dropbox/1_PROJECTS/knowledge-management
  python3 scripts/check_ingest.py "<vault path>" --since $(date +%Y-%m-%d)
It must exit clean. Fix what it reports and run it again. Then commit and push, and
update INGEST-LEDGER.md by recomputing the counts, not by editing the numbers by hand.

EVERY THIRD BATCH
Get an independent Codex review covering those three batches before starting the next
one. The exact command and the required contents of the review brief are in the runbook.
Expect a "needs-rework" verdict — both reviews so far returned one, and that is the
process working, not a failure. Verify each finding against the source yourself, then
apply every HIGH and MEDIUM.

STOP AND ASK THE USER ONLY IF
- you would delete or overwrite anything inside a vault (the vaults are NOT in git, so
  this is unrecoverable — you should never need to)
- a source clearly belongs in a different vault
- two sources contradict on something load-bearing and you cannot represent both
- check_ingest.py reports a failure you cannot fix
- a vault's CLAUDE.md tells you to do something the runbook forbids
Everything else is your call: decide it, note it in the vault's Wiki/log.md, keep going.
Do not stop to ask whether a page is good enough or how to word something.

Start with the cluster marked "next" in the ledger. Tell me which batch you are starting
and which sources it covers, then proceed without waiting for approval.
```

---

## Notes for the human

- The prompt assumes the agent can run Bash and edit files in both the repo and the vault
  directory. Give it those permissions or it will stall on step 1.
- `.venv/bin/python` in the repo has the conversion dependencies. Plain `python3` is fine
  for the ingest and checker scripts.
- If you want to cap a session, tell it a batch count ("do 3 batches then stop and
  summarize"). Left alone it will keep going until the ledger is empty.
- The one thing worth spot-checking yourself: open one new summary page per batch and
  confirm a couple of its claims against the Raw file it cites. The checker verifies
  structure, never fidelity.
