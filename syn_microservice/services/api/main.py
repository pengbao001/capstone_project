import json
import os
import time
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fairlearn.datasets import fetch_diabetes_hospital
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from redis import Redis

from pydantic import BaseModel, Field
from llm_service import ollama_chat, ollama_json

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "/artifacts"))
TARGET = "readmit_binary"
DATA_PATH = ARTIFACT_DIR / "fairlearn_diabetes_hospital.pkl"
SENSITIVE_COLUMNS = ["race", "gender"]

redis = Redis.from_url(REDIS_URL, decode_responses=True)
app = FastAPI(title="Synthetic Data API")


class PlanRequest(BaseModel):
    user_goal : str = Field(..., description="User goal for the plan")

class RunPlan(BaseModel):
    dataset : str = 'diabetes'
    generators : list[str]
    num_rows : int
    subset_rows : int
    pair_metric_rows : int
    privacy_max_n : int
    privacy_percentile : float
    save_plots : bool = True

class ExplainRequest(BaseModel):
    run_id : str
    model_name : str

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "dataset": {"type": "string"},
        "generators": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["gaussian_copula", "ctgan", "tvae"],
            },
            "minItems": 1,
        },
        "num_rows": {"type": "integer", "minimum": 100, "maximum": 200000},
        "subset_rows": {"type": "integer", "minimum": 0, "maximum": 100000},
        "pair_metric_rows": {"type": "integer", "minimum": 500, "maximum": 20000},
        "privacy_max_n": {"type": "integer", "minimum": 500, "maximum": 20000},
        "privacy_percentile": {"type": "number", "minimum": 0.1, "maximum": 10.0},
        "save_plots": {"type": "boolean"},
    },
    "required": [
        "dataset",
        "generators",
        "num_rows",
        "subset_rows",
        "pair_metric_rows",
        "privacy_max_n",
        "privacy_percentile",
        "save_plots",
    ],
    "additionalProperties": False,
}


def run_key(run_id: str) -> str:
    return f"run:{run_id}"


@lru_cache(maxsize=1)
def builtin_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Local dataset file not found: {DATA_PATH}. "
        )

    df = pd.read_pickle(DATA_PATH).copy()

    if TARGET not in df.columns:
        raise ValueError(
            f"Local dataset is missing required target column '{TARGET}'. "
        )
    return df.reset_index(drop=True)


def preview_records(df: pd.DataFrame, rows: int) -> list[dict]:
    return json.loads(df.head(rows).to_json(orient="records", date_format="iso"))


def infer_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def get_run_dir(run_id: str) -> Path:
    path = (ARTIFACT_DIR / run_id).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="run not found")
    return path


def load_results(run_id: str) -> dict:
    metrics_path = get_run_dir(run_id) / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="results not ready")
    return json.loads(metrics_path.read_text())


def relative_path(run_id: str, value: str | None) -> str | None:
    if not value:
        return None

    path = Path(value)
    if path.is_absolute():
        path = path.resolve().relative_to(get_run_dir(run_id))

    return str(path)


def existing_relative_path(run_id: str, value: str | None) -> str | None:
    rel_path = relative_path(run_id, value)
    if not rel_path:
        return None

    full_path = get_run_dir(run_id) / rel_path
    if not full_path.exists():
        return None

    return rel_path


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/dataset/preview")
def dataset_preview(rows: int = 100) -> dict:
    df = builtin_dataset()
    return {
        "meta": {
            "dataset_name": "fairlearn_diabetes_hospital",
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "target_column": TARGET,
            "sensitive_columns": SENSITIVE_COLUMNS,
        },
        "columns": [{"column": c, "kind": infer_kind(df[c])} for c in df.columns],
        "preview": preview_records(df, rows=max(1, min(rows, 200))),
        "target_distribution": {
            str(k): float(v)
            for k, v in df[TARGET].value_counts(normalize=True).sort_index().items()
        },
    }


@app.post("/runs")
def create_run(payload: dict) -> dict:
    run_id = str(uuid4())
    job = {"run_id": run_id, **payload}
    if "dataset" not in job:
        job["dataset"] = "fairlearn_diabetes_hospital"

    run_dir = ARTIFACT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(job, indent=2))

    redis.hset(
        run_key(run_id),
        mapping={
            "status": "queued",
            "created_at": str(time.time()),
            "models_done": "0",
            "progress": "0",
        },
    )
    redis.rpush("jobs", json.dumps(job))
    return {"run_id": run_id}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    data = redis.hgetall(run_key(run_id))
    if not data:
        raise HTTPException(status_code=404, detail="run not found")
    return data


@app.get("/runs/{run_id}/results")
def get_results(run_id: str) -> dict:
    return load_results(run_id)


@app.get("/runs/{run_id}/artifact")
def get_artifact(run_id: str, path: str = Query(...)):
    run_dir = get_run_dir(run_id)
    artifact_path = (run_dir / path).resolve()

    if not str(artifact_path).startswith(str(run_dir)):
        raise HTTPException(status_code=400, detail="artifact path outside run directory")
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    return FileResponse(artifact_path)


@app.get("/runs/{run_id}/models/{model_name}/trend-manifest")
def get_trend_manifest(run_id: str, model_name: str) -> dict:
    results = load_results(run_id)
    models = results.get("full", {}).get("models", {})

    if model_name not in models:
        raise HTTPException(status_code=404, detail="model not found")

    model = models[model_name]
    artifacts = model.get("artifacts", {})

    return {
        "details": {
            "individual_column_metrics_csv": existing_relative_path(run_id, artifacts.get("individual_column_metrics_csv")),
            "pair_corr_csv": existing_relative_path(run_id, artifacts.get("pair_corr_csv")),
            "pair_contingency_csv": existing_relative_path(run_id, artifacts.get("pair_contingency_csv")),
            "diag_data_validity_csv": existing_relative_path(run_id, f"reports/details/{model_name}_diag_data_validity.csv"),
            "qual_column_shapes_csv": existing_relative_path(run_id, f"reports/details/{model_name}_qual_column_shapes.csv"),
            "qual_column_pair_trends_csv": existing_relative_path(run_id, f"reports/details/{model_name}_qual_column_pair_trends.csv"),
        },
        "plots": {
            "diag_data_validity_html": existing_relative_path(run_id, f"reports/plots/{model_name}_diag_data_validity.html"),
            "qual_column_shapes_html": existing_relative_path(run_id, f"reports/plots/{model_name}_qual_column_shapes.html"),
            "qual_column_pair_trends_html": existing_relative_path(run_id, f"reports/plots/{model_name}_qual_column_pair_trends.html"),
            "column_race_html": existing_relative_path(run_id, f"reports/plots/{model_name}_column_race.html"),
            "column_gender_html": existing_relative_path(run_id, f"reports/plots/{model_name}_column_gender.html"),
            "pair_time_vs_meds_html": existing_relative_path(run_id, f"reports/plots/{model_name}_pair_time_vs_meds.html"),
        },
        "fairness": {
            "race_csv": existing_relative_path(run_id, f"fairness/{model_name}_bygroup_race.csv"),
            "gender_csv": existing_relative_path(run_id, f"fairness/{model_name}_bygroup_gender.csv"),
            "intersection_csv": existing_relative_path(run_id, f"fairness/{model_name}_bygroup_intersection.csv"),
        },
        "privacy": {
            "privacy_json": existing_relative_path(run_id, artifacts.get("privacy_json")),
        },
        "model": model,
    }

@app.post("/llm/plan")
def llm_plan(req : PlanRequest) -> dict:
    messages = [
        {
            "role" : "system",
            "content" : (
                "You are helping configure a synthetic-data experiment. "
                "Return only JSON matching the provided schema. "
                "Choose conservative, reasonable defaults. "
                "Available generators: gaussian_copula, ctgan, tvae. "
                "Use dataset='diabetes'. "
            ),
        },
        {
            "role" : "user",
            "content" : req.user_goal,
        },
    ]

    raw_plan = ollama_json(messages, PLAN_SCHEMA, temperature=0.0)
    plan = RunPlan(**raw_plan)
    return {
        "plan" : plan.model_dump()
    }

@app.post("/llm/explain")
def llm_explain(req : ExplainRequest) -> dict:
    results = load_results(req.run_id)
    models = results.get("full", {}).get("models", {})
    baseline = results.get("full", {}).get("baseline", {})

    if req.model_name not in models:
        raise HTTPException(status_code=404, detail="model not found")

    model_payload = models[req.model_name]

    prompt = (
        "Explain these experiment results for a graduate student. "
        "Use only the numbers provided. "
        "Explain: fidelity, privacy, downstream utility, add fairness. "
        "Then give one recommendation for what to try next.\n\n"
        f"Baseline:\n{json.dumps(baseline, indent=2)}\n\n"
        f"Model:\n{json.dumps(model_payload, indent=2)}\n\n"
    )

    text = ollama_chat(
        [
            {
                "role" : "system",
                "content" : (
                    "You are a careful research assistant. "
                    "Do not invent numbers. "
                    "If something is missing, say it is missing. "
                ),
            },
            {
                "role" : "user",
                "content" : prompt,
            },
        ],
        temperature=0.2,
    )

    return {
        "text" : text
    }