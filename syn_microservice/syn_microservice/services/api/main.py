import os, json, time, math
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from redis import Redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "/artifacts"))

r = Redis.from_url(REDIS_URL, decode_responses=True)
app = FastAPI()


def run_key(run_id: str) -> str:
    return f"run:{run_id}"


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    return obj


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/runs")
def create_run(payload: Dict[str, Any]):
    run_id = str(uuid4())
    job = {"run_id": run_id, **payload}

    r.hset(
        run_key(run_id),
        mapping={
            "status": "queued",
            "created_at": str(time.time()),
            "job_json": json.dumps(job),
        },
    )

    out_dir = ARTIFACT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(job, indent=2))

    r.rpush("jobs", json.dumps(job))
    return {"run_id": run_id}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    data = r.hgetall(run_key(run_id))
    if not data:
        raise HTTPException(status_code=404, detail="run not found")
    return data


@app.get("/runs/{run_id}/results")
def get_results(run_id: str):
    out_dir = ARTIFACT_DIR / run_id
    metrics_path = out_dir / "metrics.json"

    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="results not ready")

    try:
        data = json.loads(metrics_path.read_text())
    except JSONDecodeError:
        raise HTTPException(status_code=409, detail="results are being written; try again")

    return JSONResponse(content=json_sanitize(data))
