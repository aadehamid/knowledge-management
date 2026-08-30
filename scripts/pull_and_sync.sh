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
LOCK="/tmp/knowledge-sync.lock"   # lock directory appended with .d below
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# --- Prevent overlapping runs (launchd retries + manual runs) ---
# Atomic acquisition: mkdir either succeeds (we own the lock) or fails
# (someone else does) — no check-then-write race. A pid file inside records
# the owner; stale locks (>30 min or dead PID) are removed before retrying.
LOCK="$LOCK.d"
acquire_lock() {
    if mkdir "$LOCK" 2>/dev/null; then
        echo $$ > "$LOCK/pid"
        trap 'rm -rf "$LOCK"' EXIT
        return 0
    fi
    # Lock held. Verify it is genuinely live; steal stale ones.
    if [ -f "$LOCK/pid" ]; then
        LOCK_PID=$(cat "$LOCK/pid" 2>/dev/null)
        if [ -n "$LOCK_PID" ] && ! kill -0 "$LOCK_PID" 2>/dev/null; then
            echo "[$TIMESTAMP] Stale lock (pid $LOCK_PID dead) — removing and proceeding." >> "$LOG"
            rm -rf "$LOCK"
            if mkdir "$LOCK" 2>/dev/null; then
                echo $$ > "$LOCK/pid"
                trap 'rm -rf "$LOCK"' EXIT
                return 0
            fi
        fi
    fi
    LOCK_AGE=$(( $(date +%s) - $(stat -f %m "$LOCK") ))
    if [ "$LOCK_AGE" -gt 1800 ]; then
        echo "[$TIMESTAMP] Stale lock (>${LOCK_AGE}s) — removing and proceeding." >> "$LOG"
        rm -rf "$LOCK"
        if mkdir "$LOCK" 2>/dev/null; then
            echo $$ > "$LOCK/pid"
            trap 'rm -rf "$LOCK"' EXIT
            return 0
        fi
    fi
    echo "[$TIMESTAMP] Another sync is running (lock age ${LOCK_AGE}s) — exiting." >> "$LOG"
    exit 0
}
acquire_lock

echo "" >> "$LOG"
echo "========================================" >> "$LOG"
echo "[$TIMESTAMP] Sync started" >> "$LOG"
echo "========================================" >> "$LOG"

# Step 0: NotebookLM auth pre-flight.
# nlm auth is browser-cookie based and periodically expires. When it does,
# every source add burns a 401-timeout and the failure is buried in this log.
# Instead: probe once with a cheap command. If it fails, notify the user via
# macOS notification and set NLM_SKIP=1 so sync_to_vault.py skips all nlm
# calls (vault sync proceeds normally — it doesn't depend on nlm).
# Note: macOS has no GNU timeout(1); use perl alarm as a portable timeout.
unset NLM_SKIP
if command -v nlm >/dev/null 2>&1; then
    if NLM_PROBE=$(perl -e 'alarm 20; exec @ARGV or die' nlm notebook list 2>&1); then
        echo "[$TIMESTAMP] NLM auth OK — NotebookLM sync enabled." >> "$LOG"
    else
        export NLM_SKIP=1
        echo "[$TIMESTAMP] ⚠️  NLM auth EXPIRED (or nlm hung >20s) — NotebookLM sync skipped. Run 'nlm auth login' to fix." >> "$LOG"
        osascript -e 'display notification "NotebookLM auth expired — run: nlm auth login (vault sync continues)" with title "Knowledge sync: action needed" sound name "Basso"' >/dev/null 2>&1 || true
    fi
else
    export NLM_SKIP=1
    echo "[$TIMESTAMP] ⚠️  nlm CLI not found — NotebookLM sync skipped." >> "$LOG"
    osascript -e 'display notification "nlm CLI not found — NotebookLM sync skipped" with title "Knowledge sync: action needed"' >/dev/null 2>&1 || true
fi

# Step 1: Pull latest from GitHub
echo "[$TIMESTAMP] Step 1: Pulling from GitHub..." >> "$LOG"
PULL_OUTPUT=$(/usr/bin/git -C "$REPO_DIR" pull origin main 2>&1)
PULL_RC=$?
echo "$PULL_OUTPUT" >> "$LOG"

if [ "$PULL_RC" -ne 0 ]; then
    PULL_STATUS="git pull FAILED (rc=$PULL_RC) — sync aborted to avoid publishing stale content"
    echo "[$TIMESTAMP] Result: $PULL_STATUS" >> "$LOG"
    osascript -e 'display notification "git pull failed — knowledge sync aborted" with title "Knowledge sync: action needed" sound name "Basso"' >/dev/null 2>&1 || true
    exit 1
fi

if echo "$PULL_OUTPUT" | grep -q "Already up to date"; then
    PULL_STATUS="No new changes from GitHub"
else
    PULL_STATUS="New changes pulled from GitHub"
fi
echo "[$TIMESTAMP] Result: $PULL_STATUS" >> "$LOG"

# Step 2: Sync to Obsidian vault
echo "" >> "$LOG"
echo "[$TIMESTAMP] Step 2: Syncing to Obsidian vault..." >> "$LOG"
# Prefer the repo virtualenv (created by `uv venv`); fall back to system python3.
PYBIN="$REPO_DIR/.venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="$(command -v python3)"
SYNC_OUTPUT=$("$PYBIN" "$REPO_DIR/scripts/sync_to_vault.py" 2>&1)
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