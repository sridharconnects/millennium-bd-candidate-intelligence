#!/usr/bin/env bash
# Waits for a real Anthropic key to appear in .env, then runs preflight + the full
# parse. Exits as soon as the work is done (or fails), so it fires exactly one
# completion notification rather than polling forever.
cd "$(dirname "$0")/.."
LOG=/tmp/mm_pipeline.log
: > "$LOG"

echo "waiting for a key in .env …" | tee -a "$LOG"
for _ in $(seq 1 720); do          # up to ~60 minutes at 5s
  if [ -f .env ] && grep -qE '^ANTHROPIC_API_KEY=sk-ant-[A-Za-z0-9_-]{20,}' .env; then
    echo "key detected" | tee -a "$LOG"
    break
  fi
  sleep 5
done

if ! grep -qE '^ANTHROPIC_API_KEY=sk-ant-[A-Za-z0-9_-]{20,}' .env 2>/dev/null; then
  echo "TIMEOUT: no valid key appeared in .env" | tee -a "$LOG"
  exit 3
fi

# Parse live for this run; the cache it writes makes every later run offline+free.
sed -i '' 's/^DEMO_MODE=1/DEMO_MODE=0/' .env 2>/dev/null || true

echo "── preflight ──" | tee -a "$LOG"
if ! python3 scripts/preflight.py 2>&1 | tee -a "$LOG"; then
  echo "PREFLIGHT FAILED" | tee -a "$LOG"
  exit 4
fi

echo "── parsing 10 resumes ──" | tee -a "$LOG"
python3 scripts/run_pipeline.py 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

# Restore replay mode so the app, notebook and tests stay deterministic and free.
sed -i '' 's/^DEMO_MODE=0/DEMO_MODE=1/' .env 2>/dev/null || true

if [ "$rc" -ne 0 ]; then
  echo "PIPELINE FAILED rc=$rc" | tee -a "$LOG"
  exit "$rc"
fi
echo "PIPELINE OK — $(ls data/llm_cache/*.json 2>/dev/null | wc -l | tr -d ' ') cache entries written" | tee -a "$LOG"
