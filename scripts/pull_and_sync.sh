#!/bin/bash
# Pull latest from GitHub and sync markdown + images to Obsidian vault.
# Used by launchd to run daily (catches up on missed runs on wake).
#
# PATH note: launchd runs this with a minimal PATH (/usr/bin:/bin:...).
# The nlm CLI (NotebookLM sync) lives in ~/.local/bin and git in
# /usr/local/bin (or /opt/homebrew/bin on Apple Silicon), so we extend
# PATH here explicitly. Without this, sync_to_vault.py reports
# "nlm CLI not found — skipping NotebookLM sync" even though nlm works
# fine in an interactive shell.

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

REPO_DIR="/Users/hamidadesokan/Dropbox/1_PROJECTS/knowledge-management"
LOG="$HOME/.knowledge-sync.log"
LOCK="/tmp/knowledge-sync.lock"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# --- Prevent overlapping runs (launchd retries + manual runs) ---
if [ -f "$LOCK" ]; then
    LOCK_AGE=$(( $(date +%s) - $(stat -f %m "$LOCK") ))
    if [ "$LOCK_AGE" -gt 1800 ]; then
        echo "[$TIMESTAMP] Stale lock (>${LOCK_AGE}s) — removing and proceeding." >> "$LOG"
        rm -f "$LOCK"
    else
        echo "[$TIMESTAMP] Another sync is running (lock age ${LOCK_AGE}s) — exiting." >> "$LOG"
        exit 0
    fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "" >> "$LOG"
echo "========================================" >> "$LOG"
echo "[$TIMESTAMP] Sync started" >> "$LOG"
echo "========================================" >> "$LOG"

# Step 1: Pull latest from GitHub
echo "[$TIMESTAMP] Step 1: Pulling from GitHub..." >> "$LOG"
PULL_OUTPUT=$(/usr/bin/git -C "$REPO_DIR" pull origin main 2>&1)
echo "$PULL_OUTPUT" >> "$LOG"

if echo "$PULL_OUTPUT" | grep -q "Already up to date"; then
    PULL_STATUS="No new changes from GitHub"
else
    PULL_STATUS="New changes pulled from GitHub"
fi
echo "[$TIMESTAMP] Result: $PULL_STATUS" >> "$LOG"

# Step 2: Sync to Obsidian vault
echo "" >> "$LOG"
echo "[$TIMESTAMP] Step 2: Syncing to Obsidian vault..." >> "$LOG"
SYNC_OUTPUT=$(python3 "$REPO_DIR/scripts/sync_to_vault.py" 2>&1)
echo "$SYNC_OUTPUT" >> "$LOG"

if echo "$SYNC_OUTPUT" | grep -q "0 file(s) synced"; then
    SYNC_STATUS="No new files to sync — vault is up to date"
else
    SYNC_STATUS="New files synced to Obsidian vault"
fi
echo "[$TIMESTAMP] Result: $SYNC_STATUS" >> "$LOG"

# Summary
echo "" >> "$LOG"
echo "[$TIMESTAMP] Summary: $PULL_STATUS | $SYNC_STATUS" >> "$LOG"
echo "========================================" >> "$LOG"