#!/bin/sh
set -eu

# Wait for Redis to become reachable.
python - <<'PY'
import os, time
from redis import Redis

url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
r = Redis.from_url(url, decode_responses=True)

for _ in range(60):  # ~2 minutes
    try:
        r.ping()
        break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("Redis not reachable; check REDIS_URL/connectivity.")
PY

# Start the background worker (consumes jobs from Redis).
python -u /app/worker.py &

# Start the API server (foreground keeps the container alive).
# Health endpoint is `/health` (see `services/api/main.py`).
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}