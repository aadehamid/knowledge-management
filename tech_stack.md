# Tech Stack

- **NotebookLM sync**: [nlm CLI](https://github.com/tmc/nlm) — pushes converted markdown sources to NotebookLM notebooks. Auth via `nlm auth`; notebook IDs in `sync_config.json`.
- **Obsidian vault sync**: `scripts/sync_to_vault.py` — copies converted markdown + images to the vault as plain files (Obsidian reads the folder directly; no CLI required).
- **Conversion**: cloud agent runs `scripts/convert_pdfs.py` (pymupdf4llm / markitdown) → PR on `auto/daily-conversions` → CI reviewer merges if clean.
- **Local automation**: `scripts/pull_and_sync.sh` via launchd (`com.hamid.knowledge-sync.plist`) — daily `git pull` + vault sync + NotebookLM push.
