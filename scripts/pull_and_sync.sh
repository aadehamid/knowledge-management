#!/bin/bash
# Pull latest from GitHub and sync markdown + images to Obsidian vault.
# Used by launchd to run daily (catches up on missed runs on wake).

REPO_DIR="/Users/hamidadesokan/Dropbox/1_PROJECTS/knowledge-management"
LOG="$HOME/.knowledge-sync.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "" >> "$LOG"
echo "========================================" >> "$LOG"
echo "[$TIMESTAMP] Sync started" >> "$LOG"
echo "========================================" >> "$LOG"

# Step 1: Pull latest from GitHub
echo "[$TIMESTAMP] Step 1: Pulling from GitHub..." >> "$LOG"
PULL_OUTPUT=$(/usr/bin/git -C "$REPO_DIR" pull origin main 2>&1)
echo "$PULL_OUTPUT" >> "$LOG"

# Check if there were new changes
if echo "$PULL_OUTPUT" | grep -q "Already up to date"; then
    PULL_STATUS="No new changes from GitHub"
else
    PULL_STATUS="New changes pulled from GitHub"
fi
echo "[$TIMESTAMP] Result: $PULL_STATUS" >> "$LOG"

# Step 2: Sync to Obsidian vault
echo "" >> "$LOG"
echo "[$TIMESTAMP] Step 2: Syncing to Obsidian vault..." >> "$LOG"
SYNC_OUTPUT=$(/usr/local/bin/python3 "$REPO_DIR/scripts/sync_to_vault.py" 2>&1)
echo "$SYNC_OUTPUT" >> "$LOG"

# Check if anything was synced
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
