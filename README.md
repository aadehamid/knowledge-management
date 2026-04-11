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

3. Commit, push, and follow steps 3-5 from above.

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
