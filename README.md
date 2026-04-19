# Knowledge Management

A personal knowledge management repository for organizing notes, references, and insights. Includes an automated pipeline for converting PDFs and web pages to markdown, organized by subject area, with sync to Obsidian.

## Structure

```
knowledge-management/
├── notes/                        # Atomic notes, fleeting thoughts, and evergreen content
├── references/
│   └── papers/                   # Converted markdown output, organized by subject
│       ├── transformers/
│       │   ├── 1706.03762.md
│       │   └── 1706.03762_images/
│       └── cuda/
├── resources/
│   └── sources/                  # Input sources, organized by subject
│       ├── transformers/
│       │   └── urls.txt
│       └── cuda/
│           └── urls.txt
├── scripts/
│   ├── convert_pdfs.py           # Cloud agent conversion script
│   └── sync_to_vault.py          # Local Obsidian vault sync script
├── projects/                     # Project-specific knowledge and documentation
├── templates/                    # Reusable document and note templates
├── sync_config.json              # Subject → Obsidian vault path mapping
├── requirements.txt              # Python dependencies (pymupdf4llm, markitdown)
└── README.md
```

## PDF & Web Page Conversion Pipeline

The pipeline converts PDFs and web pages to markdown, extracts images from PDFs, and syncs the output to your Obsidian vault with proper frontmatter for the wiki schema.

### How it works

1. **Add source URLs** to `resources/sources/<subject>/urls.txt`
2. **Cloud agent** downloads PDFs, converts all sources to markdown (with images), and opens a PR
3. **Merge the PR** and `git pull` locally
4. **Sync to Obsidian** with `python scripts/sync_to_vault.py`

The sync script renames files to the wiki schema (`<source_type>-<slug>.md`), prepends YAML frontmatter, and copies extracted images alongside the markdown.

### urls.txt format

Each line follows the format: `url | title | source_type`

```
https://arxiv.org/pdf/1706.03762.pdf | Attention Is All You Need | paper
https://docs.nvidia.com/cuda/cuda-programming-guide/index.html | CUDA Programming Guide | doc
```

Valid source types: `paper`, `blog`, `video`, `course`, `code`, `thread`, `pdf`, `doc`

Blank lines and lines starting with `#` are ignored.

### Adding a source to an existing subject

1. Add the URL to `resources/sources/<subject>/urls.txt`
2. Commit and push
3. Run the cloud agent:
   ```sh
   oz agent run-cloud --environment 3SyIIpxdPQfIyGOFd7ZQTs --prompt "Run: cd knowledge-management && python scripts/convert_pdfs.py. If any new files were generated, create a PR."
   ```
4. Merge the PR, then `git pull`
5. Run `python scripts/sync_to_vault.py`

### Adding a new subject area

1. Create the subject folder with a `urls.txt`:
   ```sh
   mkdir -p resources/sources/<new-subject>
   ```
   Add URLs to `resources/sources/<new-subject>/urls.txt`

2. Add the Obsidian vault mapping in `sync_config.json`:
   ```json
   {
     "subjects": {
       "<new-subject>": "/path/to/obsidian/vault/Raw"
     }
   }
   ```

3. *(Optional)* Register a NotebookLM notebook for the subject so new markdown is synced to NotebookLM via the `nlm` CLI. Add the notebook ID under `notebooklm` in `sync_config.json`:
   ```json
   {
     "notebooklm": {
       "<new-subject>": "<notebook-id>"
     }
   }
   ```
   If omitted, only the NotebookLM push is skipped for this subject — the Obsidian vault sync (`sync_to_vault.py`) still runs normally.

4. Create the Obsidian vault for the subject area with the required layer folders:
   ```sh
   mkdir -p "/path/to/obsidian/vault/<new-subject>"/{Raw,Wiki,"Learning Path"}
   cp "/path/to/obsidian/vault/<existing-subject>/CLAUDE.md" "/path/to/obsidian/vault/<new-subject>/CLAUDE.md"
   ```
   Then **adapt the cloned `CLAUDE.md` to the new domain** — strip every term, example, and Learning Path stage scope that belongs to the sibling subject, and substitute domain-native terms. A schema clone is a starting point, not a drop-in.

5. **Run the vault bootstrap workflow** (defined in the vault's own `CLAUDE.md`, section "Bootstrap workflow"). This is a **mandatory one-time step** for every new subject-area vault. The bootstrap creates `Wiki/index.md`, `Wiki/log.md`, `Wiki/overview.md`, plus stub entity/concept pages for the domain's core terms, plus empty `Learning Path/` stage files. Without bootstrap, the first ingest has nothing to link into — summaries end up as orphans and the wiki graph-connectivity invariant is broken.

6. Commit, push, and follow steps 3-5 from the "Adding a source to an existing subject" section above.

### Resuming a session on any vault

When starting a new session — on any subject-area vault in this knowledge base — the LLM has no memory of prior work. Before giving any new instructions, ask it to orient itself by reading the load-bearing state files so it can pick up where the last session left off without guessing.

**The three checks, in order:**

1. **Read `Wiki/log.md`** for the vault you're working in — the append-only log tells you what's been ingested, what stubs were created, what's deferred, and what was touched last. This is the single most load-bearing file for session continuity.
2. **Check `Raw/` for unprocessed files** — anything newer than the latest log entry is pending ingest.
3. **Glance at `git status` and recent commits** — catches anything changed outside the log (config updates, `CLAUDE.md` edits, new subject areas, URL additions).

**Suggested resume prompt** (vault-agnostic — just substitute the vault path):

> Review `Wiki/log.md` in the `<vault-name>` vault, check `Raw/` for unprocessed files, glance at `git status` / recent commits, and tell me what state we're in before we start. Don't edit anything yet.

If you're working across multiple vaults in one session (e.g., bouncing between the Transformer and CUDA vaults), ask for the orientation check on each vault explicitly — the LLM will only check the vault you name.

The vault's own `CLAUDE.md` contains a longer-form version of this resume prompt tailored to that vault. Use the short form for daily interactive sessions; reach for the long form only when the vault is unfamiliar or it's been weeks since the last session.

### Converters used

- **PDFs**: `pymupdf4llm` — lightweight, extracts images inline at their original position
- **Web pages**: `markitdown` (Microsoft) — converts HTML URLs directly to markdown

### Warp cloud environment

- **Environment ID**: `3SyIIpxdPQfIyGOFd7ZQTs`
- **Docker image**: `warpdotdev/dev-base:latest-agents`
- **Setup command**: `cd knowledge-management && pip install --break-system-packages -r requirements.txt`

## Getting Started

Clone the repository and install dependencies:

```sh
git clone https://github.com/aadehamid/knowledge-management.git
pip install -r requirements.txt
```
