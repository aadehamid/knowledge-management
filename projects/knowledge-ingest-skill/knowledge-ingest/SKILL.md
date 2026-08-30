---
name: knowledge-ingest
description: Turn fetched source documents into an interlinked, provenance-checked knowledge bundle — wiki pages, concept pages and a curriculum — with an independent cross-model review gate. Use this whenever a corpus of raw sources needs to become durable knowledge: ingesting a batch into a vault or bundle, resuming a partially-processed backlog, routing a fresh list of URLs into subject areas, bootstrapping a new subject bundle, or auditing an existing one for provenance, dead links and consistency. Works at any size — a single new article added to an existing vault, or a backlog of hundreds. Use it even when the user just says "process the rest of these sources", "add this one to the wiki", or "pick up where we left off" without naming a method. Not for one-off summarisation of a single document, not for fetching or converting sources into markdown, and not for answering questions from a knowledge base that already exists.
---

# Knowledge ingest

You already know how to read sources and write linked pages. This skill is not about that.
It is about the four ways that work goes quietly wrong, and the two gates that catch them.

The failure mode is silent: a page that reads beautifully and says something its source
never said is indistinguishable from a good page until someone checks.

## Configuration

Read `knowledge-ingest.config.json` from the project root — bundles with their paths and
shapes, the validator command, the reviewer command, the ledger. Never hardcode a path; if
you are typing one, the project should be describing it instead. With no config, ask for
the bundle and source paths, then offer to write one.

Orient before acting: read the ledger and the bundle's own schema file, and **recompute the
counts rather than trusting the written numbers**. A source counts as processed when some
page in the bundle cites its filename — frontmatter flags drift, the citation is the fact.

---

## The four invariants

### 1. Every factual sentence traces to a line in a source *for this batch*

Not "true". Not "well known". **In the source, in this batch.** Outside knowledge is
invention however correct it happens to be — the reader's trust comes from the provenance
chain, and a true-but-unsourced sentence breaks it as thoroughly as a false one.

A page once read "FSDP shards parameters, gradients and optimizer state." Accurate, and
`optimizer state` appeared **zero times** in the four permitted sources. Self-checking
never caught it; an independent reviewer did.

The test: *would I write this sentence without the source open?* If yes, delete it or go
find the line.

### 2. Hedges belong to the claim

*Likely*, *suggests*, *on this dataset*, *assuming X*, *we found the most support for* —
load-bearing, all of them. Dropping them converts a researcher's careful hypothesis into
your confident assertion, and the reader cannot tell.

Written "the gap occurs **because** of BA dynamics"; the source said "the **likely reason**
is ... **on this dataset**." Written "the paper's **answer** is the generation-verification
gap"; the paper "found the **most support** for" it among hypotheses it tested.

### 3. Record disagreements; never resolve them

Two sources conflicting marks where the field is unsettled — the most valuable thing a
corpus holds. Present both, name the sources, say plainly that they differ. If one has
better evidence, say why *and* keep the other.

Resist synthesis. A reconciliation you invented, sitting in a page whose frontmatter cites
two sources, reads exactly like something those sources said. If you must relate them, mark
the relation as your inference rather than their finding.

### 4. Verify against the system before acting on it

Titles mislead, keyword matches mislead, plans go stale while you execute them.

A source titled "build a ChatGPT" was about to be moved out of an inference bundle as
off-topic; one grep showed the bundle's core `KV Cache` page cited it as the reference
implementation. Before relocating, deleting or reclassifying: grep for inbound references,
then ask what could undo the change — deleting a page whose source line still exists means
the next sync restores it.

---

## The two gates

### Gate A — mechanical, every batch

`scripts/check_ingest.py <bundle> --validator <cmd>` checks schema, links, reference
reciprocity in both directions, trust stamps, stale markers, index coverage, log ordering.

**A clean run means the structure is sound and nothing more.** It cannot tell you whether a
sentence is in its source. Reading "Clean" as "correct" is the most likely way this skill
fails in practice — and read the output for *skipped* checks, because a warning is not a
pass.

### Gate B — independent review

**A different agent instance, ideally a different model family, report-only.** Never
self-review: the agent that wrote a page cannot see what it invented, because the invented
part felt like knowing. Where the runtime can enforce read-only, prefer that to an
instruction.

The brief needs the schema, **every source labelled as the only permitted evidence**, the
explicit file list, the check priorities, and — the highest-yield line in the whole
procedure — **the claims you are least sure of, named, with a request to attack them.**
Reviewers catch what they are pointed at.

You own resolution, not the reviewer: verify each finding against the source before
applying it, document overrides rather than silently dropping them, apply every HIGH and
MEDIUM, re-run Gate A, and record which reviewer was used and what it found.

Expect `needs-rework`. A review that finds nothing is more likely a review that was not
given the sources. Details: `references/review.md`.

---

## Scale the ceremony, not the invariants

One source and eighty are the same job at different sizes. The invariants hold identically
— a single page with an unsourced claim is exactly as wrong as one in a batch of forty.
Only the machinery scales.

**One or two sources**: no cluster selection, no ledger entry — there is no cluster and
nothing to resume. Read, write, wire it in (index, back-reference, log line), run Gate A.
Gate B optional; note in the log that it was unreviewed so nobody later assumes otherwise.

**Three to seven**: the above plus a log entry naming the group. Review if the sources are
dense or the claims quantitative.

**Eight or more**: the full loop — cluster selection, ledger, both gates. Procedure and page
templates in `references/ingest.md`.

If following this skill makes a one-source ingest feel heavy, it is being applied wrongly.

---

## Traps

Non-obvious, and each cost real time:

| Trap | What happens |
|---|---|
| Filename collisions | Stem functions are rarely injective and `Path.stem` truncates dotted names — `llama3.3`→`llama3`, every arXiv `2401.*`→`2401`. The second file overwrites the first and inherits its backlinks. |
| Unquoted YAML colons | `description: Foo: bar` parses as a mapping and fails validation. |
| Guessed timestamps | A stamp dated after the file's own mtime is worse than none. |
| Silent converter degradation | A missing optional dependency turned every video into a 200-word shell instead of a 13,000-word transcript. Plausible output, 98% of the content gone. Assert the expected shape. |
| URL identity | The same article lives at many URLs — host moves, path renames, repo renames, `/live/` vs `?v=`. A naive diff reported 123 missing where 64 was true. Use `scripts/url_identity.py`. |

---

## Modes

- **route** a URL list into subjects: `references/route.md`. Two things carry it — never
  classify by keyword (a regex classifier built for exactly this sent Unsloth to inference
  and backpropagation to transformers), and make any new subject **earn its place**. The
  default home is an existing bundle; a new one must show that no current bundle covers the
  material, that there is enough of it to sustain the scaffolding, and that it is a domain
  rather than a feed. Name it from what the corpus contains, not from what the list was
  called.
- **bootstrap** a new bundle: `references/bootstrap.md`. Choose concept or catalog shape
  deliberately, and derive stub pages from the corpus's own term frequencies rather than a
  template — cloned stub lists produce pages nobody fills.
- **audit** an existing bundle: `references/audit.md`. Sample for fidelity and report an
  error *rate*; structure alone tells you nothing about accuracy.

## Stop and ask

Only these — an agent that halts on every judgement call never finishes a backlog:

- You would delete or overwrite anything not version-controlled.
- A source clearly belongs to a different bundle.
- Two sources contradict on something load-bearing and you cannot represent both.
- Gate A reports a failure you cannot fix.
- A bundle's own schema contradicts this skill.

Everything else is yours. Decide it, record it in the log, keep going.
