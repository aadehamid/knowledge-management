#!/usr/bin/env python3
"""DeepSeek V4 Pro 0813 review companion for the knowledge-base ingest.

Replaces the Codex companion as the independent reviewer (user direction,
2026-08-30). Uses the Nous Portal inference API (already authenticated by
`hermes auth`), model deepseek/deepseek-v4-pro-0813.

Usage:
  python3 scripts/deepseek_review.py submit "<brief-file>"
      -> prints {"job_id": ...}
  python3 scripts/deepseek_review.py status <job-id>
      -> prints {"status": "running"|"done", ...}
  python3 scripts/deepseek_review.py result <job-id>
      -> prints the full review text

Jobs run in the background (the review reads every Raw source and can take
10-30 minutes); state is stored under projects/llm-corpus-expansion/.reviews/.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JOBS = REPO / "projects" / "llm-corpus-expansion" / ".review-jobs"
JOBS.mkdir(parents=True, exist_ok=True)

MODEL = "deepseek/deepseek-v4-pro-0813"
BASE = "https://inference-api.nousresearch.com/v1"


def _token() -> str:
    auth = json.loads((Path.home() / ".hermes" / "auth.json").read_text())
    return auth["providers"]["nous"]["access_token"]


def _call(messages: list, max_tokens: int = 16384) -> str:
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps({
            "model": MODEL,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": messages,
        }).encode(),
        headers={
            "Authorization": f"Bearer {_token()}",
            "Content-Type": "application/json",
            "User-Agent": "hermes-agent",  # the API rejects requests without a UA
        },
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        body = json.loads(r.read())
    return body["choices"][0]["message"]["content"]


def submit(brief_file: str) -> str:
    brief = Path(brief_file).read_text(encoding="utf-8")
    job_id = time.strftime("%Y%m%d-%H%M%S")
    (JOBS / f"{job_id}.brief.md").write_text(brief, encoding="utf-8")
    (JOBS / f"{job_id}.status").write_text("running", encoding="utf-8")
    # Detach: run the call in a forked child so the CLI returns immediately.
    pid = os.fork()
    if pid == 0:
        try:
            result = _call([
                {"role": "system", "content": (
                    "You are an independent, adversarial reviewer of knowledge-base "
                    "ingest work. Report findings, do NOT edit any file. Be exact: "
                    "every finding needs file, line, and the source text that "
                    "contradicts or supports it. Output findings tagged HIGH / MEDIUM "
                    "/ LOW, then one verdict line: ship | ship-with-fixes | needs-rework."
                )},
                {"role": "user", "content": brief},
            ])
            (JOBS / f"{job_id}.result.md").write_text(result, encoding="utf-8")
            (JOBS / f"{job_id}.status").write_text("done", encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            (JOBS / f"{job_id}.error").write_text(str(e), encoding="utf-8")
            (JOBS / f"{job_id}.status").write_text("error", encoding="utf-8")
        os._exit(0)
    return job_id


def status(job_id: str) -> str:
    return (JOBS / f"{job_id}.status").read_text()


def result(job_id: str) -> str:
    f = JOBS / f"{job_id}.result.md"
    if f.exists():
        return f.read_text(encoding="utf-8")
    e = JOBS / f"{job_id}.error"
    if e.exists():
        return f"ERROR: {e.read_text()}"
    return "(still running)"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "submit" and len(sys.argv) > 2:
        print(json.dumps({"job_id": submit(sys.argv[2])}))
    elif cmd == "status" and len(sys.argv) > 2:
        print(json.dumps({"status": status(sys.argv[2])}))
    elif cmd == "result" and len(sys.argv) > 2:
        print(result(sys.argv[2]))
    else:
        print(__doc__)
