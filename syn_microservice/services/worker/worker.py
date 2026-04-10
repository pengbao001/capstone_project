import os, json, time, traceback, shutil
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import pickle
from redis import Redis

from fairlearn.datasets import fetch_diabetes_hospital
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    true_positive_rate,
    false_positive_rate,
    count
)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors

from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from sdmetrics.single_column import (
    KSComplement,
    TVComplement,
    CategoryCoverage,
    RangeCoverage,
    MissingValueSimilarity,
    StatisticSimilarity,
    BoundaryAdherence,
    CategoryAdherence,
)
from sdmetrics.column_pairs import CorrelationSimilarity, ContingencySimilarity
from sdmetrics.single_table import NewRowSynthesis, TableStructure

from sdmetrics.visualization import get_column_plot, get_column_pair_plot


REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "/artifacts"))
r = Redis.from_url(REDIS_URL, decode_responses=True)

TARGET = "readmit_binary"

PRETRAINED = {
    "gaussian_copula": ARTIFACT_DIR / "gaussian_copuula_diabetes.pkl",
    "ctgan": ARTIFACT_DIR / "ctgan_diabetes.pkl",
    "tvae": ARTIFACT_DIR / "tvae_diabetes.pkl",
}
META_PATH = ARTIFACT_DIR / "diabetes_metadata.json"

def json_sanitize(obj):

    if obj is None:
        return None

    if isinstance(obj, np.generic):
        return json_sanitize(obj.item())

    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None

    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]

    if isinstance(obj, (int, bool, str)):
        return obj

    return str(obj)


def atomic_write_json(path: Path, obj):
    safe = json_sanitize(obj)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(safe, indent=2, allow_nan=False))
    os.replace(tmp, path)

def run_key(run_id: str) -> str:
    return f"run:{run_id}"

def set_status(run_id: str, status: str, **extra):
    payload = {"status": status, **{k: str(v) for k, v in extra.items()}}
    r.hset(run_key(run_id), mapping=payload)

def guess_sdtype_from_series(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numerical"
    return "categorical"

def load_or_build_single_table_metadata(df: pd.DataFrame, path: Path | None = None, table_name_hint: str = "diabetes") -> dict:
    """
    Returns sdmetrics-style metadata dict:
      {"columns": {"col": {"sdtype": ...}, ...}}
    """
    if path is not None and path.exists():
        with path.open("r") as f:
            meta = json.load(f)

        if "tables" in meta:
            table_name = table_name_hint if table_name_hint in meta["tables"] else next(iter(meta["tables"]))
            meta = meta["tables"][table_name]

        if "columns" in meta:
            if "race" in meta["columns"]:
                meta["columns"]["race"]["sdtype"] = "categorical"
            if "gender" in meta["columns"]:
                meta["columns"]["gender"]["sdtype"] = "categorical"
            if TARGET in meta["columns"]:
                meta["columns"][TARGET]["sdtype"] = "boolean"
            return meta

    meta = {"columns": {}}
    for col in df.columns:
        meta["columns"][col] = {"sdtype": guess_sdtype_from_series(df[col])}

    for col in ["race", "gender"]:
        if col in meta["columns"]:
            meta["columns"][col]["sdtype"] = "categorical"
    if TARGET in meta["columns"]:
        meta["columns"][TARGET]["sdtype"] = "boolean"

    return meta

def coerce_boolean(series: pd.Series) -> pd.Series:
    s = series.copy()

    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")

    mapping = {0: False, 1: True, "0": False, "1": True, False: False, True: True}
    s2 = s.map(mapping)

    if s2.notna().mean() < 0.9 and s.notna().mean() > 0:
        s2 = pd.to_numeric(s, errors="coerce").map(lambda x: np.nan if pd.isna(x) else bool(int(x)))

    return s2.astype("boolean")

def sanitize_for_sdmetrics(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    out = df.copy()

    for col, info in meta.get("columns", {}).items():
        if col not in out.columns:
            continue
        sdtype = info.get("sdtype")

        if sdtype == "boolean":
            out[col] = coerce_boolean(out[col])
        elif sdtype in ("categorical", "id", "other"):
            out[col] = out[col].astype("object")
        elif sdtype == "numerical":
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif sdtype == "datetime":
            out[col] = pd.to_datetime(out[col], errors="coerce")

    return out

def align_like_real(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> pd.DataFrame:
    syn = syn_df.copy()

    extra = [c for c in syn.columns if c not in real_df.columns]
    if extra:
        syn = syn.drop(columns=extra)

    missing = [c for c in real_df.columns if c not in syn.columns]
    for c in missing:
        syn[c] = np.nan

    return syn[real_df.columns]


def load_diabetes(job: dict):
    data = fetch_diabetes_hospital(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()

    dropped_columns = [c for c in ["readmitted", "readmit_binary"] if c in X.columns]
    X = X.drop(columns=dropped_columns, errors="ignore")

    real_data = X.copy()
    real_data[TARGET] = (y == 1)

    test_size = float(job.get("test_size", 0.2))
    random_state = int(job.get("random_state", 66))

    real_train, real_test = train_test_split(
        real_data,
        test_size=test_size,
        random_state=random_state,
        stratify=real_data[TARGET]
    )
    real_train = real_train.reset_index(drop=True)
    real_test  = real_test.reset_index(drop=True)

    cat_cols = real_train.select_dtypes(include=["category"]).columns
    if len(cat_cols):
        real_train[cat_cols] = real_train[cat_cols].astype("object")
        real_test[cat_cols]  = real_test[cat_cols].astype("object")

    real_train[TARGET] = real_train[TARGET].astype(bool)
    real_test[TARGET]  = real_test[TARGET].astype(bool)

    subset_rows = int(job.get("subset_rows", 0))
    if subset_rows and subset_rows < len(real_train):
        pos = real_train[real_train[TARGET] == True]
        neg = real_train[real_train[TARGET] == False]
        frac = subset_rows / len(real_train)
        pos_n = max(1, int(round(len(pos) * frac)))
        neg_n = max(1, int(round(len(neg) * frac)))

        real_train = pd.concat([
            pos.sample(min(pos_n, len(pos)), random_state=random_state),
            neg.sample(min(neg_n, len(neg)), random_state=random_state),
        ]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return real_train, real_test


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing pretrained model: {model_path}")
    with model_path.open("rb") as f:
        return pickle.load(f)

def run_reports_for_one(name: str, real_df: pd.DataFrame, syn_df: pd.DataFrame, meta: dict, verbose: bool = False):
    diag = DiagnosticReport()
    diag.generate(real_df, syn_df, meta, verbose=verbose)

    qual = QualityReport()
    qual.generate(real_df, syn_df, meta, verbose=verbose)

    return diag, qual

def props_to_dict(props):
    if isinstance(props, pd.DataFrame) and "Property" in props.columns and "Score" in props.columns:
        return dict(zip(props["Property"], props["Score"]))
    if isinstance(props, list):
        try:
            return dict(props)
        except Exception:
            return {}
    if isinstance(props, dict):
        return props
    return {}

def compute_column_metric_safe(metric_cls, real_col, syn_col, **kwargs) -> float:
    try:
        return float(metric_cls.compute(real_data=real_col, synthetic_data=syn_col, **kwargs))
    except TypeError:
        try:
            return float(metric_cls.compute(real_col, syn_col, **kwargs))
        except Exception:
            return np.nan
    except Exception:
        return np.nan

def per_column_metrics(real_df: pd.DataFrame, syn_df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    rows = []
    for col, info in meta["columns"].items():
        if col not in real_df.columns or col not in syn_df.columns:
            continue

        sdtype = info.get("sdtype")
        r = real_df[col]
        s = syn_df[col]

        rows.append({
            "column": col,
            "sdtype": sdtype,
            "metric": "MissingValueSimilarity",
            "score": compute_column_metric_safe(MissingValueSimilarity, r, s),
        })

        if sdtype in ("numerical", "datetime"):
            rows.extend([
                {"column": col, "sdtype": sdtype, "metric": "KSComplement", "score": compute_column_metric_safe(KSComplement, r, s)},
                {"column": col, "sdtype": sdtype, "metric": "RangeCoverage", "score": compute_column_metric_safe(RangeCoverage, r, s)},
                {"column": col, "sdtype": sdtype, "metric": "BoundaryAdherence", "score": compute_column_metric_safe(BoundaryAdherence, r, s)},
                {"column": col, "sdtype": sdtype, "metric": "StatisticSimilarity(mean)", "score": compute_column_metric_safe(StatisticSimilarity, r, s, statistic="mean")},
                {"column": col, "sdtype": sdtype, "metric": "StatisticSimilarity(median)", "score": compute_column_metric_safe(StatisticSimilarity, r, s, statistic="median")},
                {"column": col, "sdtype": sdtype, "metric": "StatisticSimilarity(std)", "score": compute_column_metric_safe(StatisticSimilarity, r, s, statistic="std")},
            ])
        elif sdtype in ("categorical", "boolean", "id", "other"):
            rows.extend([
                {"column": col, "sdtype": sdtype, "metric": "TVComplement", "score": compute_column_metric_safe(TVComplement, r, s)},
                {"column": col, "sdtype": sdtype, "metric": "CategoryCoverage", "score": compute_column_metric_safe(CategoryCoverage, r, s)},
                {"column": col, "sdtype": sdtype, "metric": "CategoryAdherence", "score": compute_column_metric_safe(CategoryAdherence, r, s)},
            ])

    return pd.DataFrame(rows)

def infer_cols_from_metadata(meta: dict):
    num_cols, cat_cols = [], []
    for col, info in meta["columns"].items():
        sdtype = info.get("sdtype")
        if sdtype in ("numerical", "datetime"):
            num_cols.append(col)
        elif sdtype in ("categorical", "boolean", "id", "other"):
            cat_cols.append(col)
    return num_cols, cat_cols

def compute_pair_metric_safe(metric_cls, real_df2, syn_df2, **kwargs) -> float:
    try:
        return float(metric_cls.compute(real_data=real_df2, synthetic_data=syn_df2, **kwargs))
    except TypeError:
        try:
            return float(metric_cls.compute(real_df2, syn_df2, **kwargs))
        except Exception:
            return np.nan
    except Exception:
        return np.nan

def per_pair_metrics(real_df: pd.DataFrame, syn_df: pd.DataFrame, meta: dict, subsample_rows: int = 5000, random_state: int = 0):
    num_cols, cat_cols = infer_cols_from_metadata(meta)

    real_use = real_df.sample(subsample_rows, random_state=random_state) if len(real_df) > subsample_rows else real_df
    syn_use  = syn_df.sample(subsample_rows, random_state=random_state)  if len(syn_df) > subsample_rows  else syn_df

    corr_rows = []
    for a, b in combinations(num_cols, 2):
        score = compute_pair_metric_safe(CorrelationSimilarity, real_use[[a, b]], syn_use[[a, b]])
        corr_rows.append({"col_a": a, "col_b": b, "metric": "CorrelationSimilarity", "score": score})

    cont_rows = []
    for a, b in combinations(cat_cols, 2):
        score = compute_pair_metric_safe(ContingencySimilarity, real_use[[a, b]], syn_use[[a, b]])
        cont_rows.append({"col_a": a, "col_b": b, "metric": "ContingencySimilarity", "score": score})

    return pd.DataFrame(corr_rows).sort_values("score"), pd.DataFrame(cont_rows).sort_values("score")


def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category", "bool", "boolean"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

def train_predict_lr(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df.drop(columns=[TARGET], errors="ignore")
    y_train = train_df[TARGET].astype(int)

    X_test = test_df.drop(columns=[TARGET], errors="ignore")
    y_test = test_df[TARGET].astype(int)

    if y_train.nunique() < 2:
        return {"auc": np.nan, "f1": np.nan, "acc": np.nan, "y_true": y_test, "y_pred": np.zeros(len(y_test), dtype=int)}

    pre = build_preprocessor(X_train)
    clf = LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1)

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    try:
        auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() >= 2 else np.nan
    except Exception:
        auc = np.nan

    return {
        "auc": auc,
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "acc": float(accuracy_score(y_test, pred)),
        "y_true": y_test,
        "y_pred": pred,
    }

def make_intersection_group(df: pd.DataFrame) -> pd.Series:
    return df["race"].astype(str).fillna("NA") + "|" + df["gender"].astype(str).fillna("NA")

def error_rate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true != y_pred))

def fairness_view(y_true, y_pred, sensitive_features, view_name: str):
    metrics = {
        "count": count,
        "selection_rate": selection_rate,
        "accuracy": accuracy_score,
        "error_rate": error_rate,
        "TPR": true_positive_rate,
        "FPR": false_positive_rate,
    }
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    dp_diff = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features))
    dp_ratio = float(demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features))

    by = mf.by_group.reset_index()

    gaps = {}
    for col in ["selection_rate", "error_rate", "TPR", "FPR"]:
        vals = pd.to_numeric(mf.by_group[col], errors="coerce").dropna()
        gaps[f"{col}_gap"] = float(vals.max() - vals.min()) if len(vals) else np.nan

    return {
        "view": view_name,
        "dp_diff": dp_diff,
        "dp_ratio": dp_ratio,
        **gaps,
        "by_group": by,
    }

def fairness_report(y_true, y_pred, real_test: pd.DataFrame):

    if y_true is None or y_pred is None or len(y_true) == 0:
        return {"dp_max": np.nan, "race": None, "gender": None, "intersection": None}

    race = fairness_view(y_true, y_pred, real_test["race"], "race")
    gender = fairness_view(y_true, y_pred, real_test["gender"], "gender")

    inter = make_intersection_group(real_test)
    inter_rep = fairness_view(y_true, y_pred, inter, "intersection")

    dp_max = float(np.nanmax([race["dp_diff"], gender["dp_diff"], inter_rep["dp_diff"]]))

    return {
        "dp_max": dp_max,
        "race": race,
        "gender": gender,
        "intersection": inter_rep,
    }

def _row_hash_series(df: pd.DataFrame) -> pd.Series:
    tmp = df.astype(str)
    return pd.util.hash_pandas_object(tmp, index=False)

def exact_duplicate_rate(real_train: pd.DataFrame, syn_df: pd.DataFrame, cols=None, max_rows: int = 20000, random_state: int = 66) -> float:
    rt = real_train.sample(min(len(real_train), max_rows), random_state=random_state)
    sy = syn_df.sample(min(len(syn_df), max_rows), random_state=random_state)

    if cols is not None:
        rt = rt[cols]
        sy = sy[cols]

    real_hashes = set(_row_hash_series(rt).tolist())
    syn_hashes = _row_hash_series(sy).tolist()
    dup = sum(1 for h in syn_hashes if h in real_hashes)
    return float(dup / max(len(syn_hashes), 1))

def privacy_nn_leakage(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    syn_df: pd.DataFrame,
    target: str,
    max_n: int = 5000,
    percentile: float = 1.0,
    random_state: int = 66,
):
    rt = real_train.sample(min(len(real_train), max_n), random_state=random_state)
    rtest = real_test.sample(min(len(real_test), max_n), random_state=random_state)
    syn = syn_df.sample(min(len(syn_df), max_n), random_state=random_state)

    X_rt = rt.drop(columns=[target], errors="ignore")
    X_rtest = rtest.drop(columns=[target], errors="ignore")
    X_syn = syn.drop(columns=[target], errors="ignore")

    pre = build_preprocessor(X_rt)
    X_rt_enc = pre.fit_transform(X_rt)
    X_rtest_enc = pre.transform(X_rtest)
    X_syn_enc = pre.transform(X_syn)

    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn.fit(X_rt_enc)

    dist_test, _ = nn.kneighbors(X_rtest_enc)
    dist_syn, _ = nn.kneighbors(X_syn_enc)

    dist_test = dist_test.ravel()
    dist_syn = dist_syn.ravel()

    tau = float(np.percentile(dist_test, percentile))
    leakage_rate = float(np.mean(dist_syn <= tau))

    return {
        "nn_percentile": float(percentile),
        "nn_tau": tau,
        "nn_leakage_rate": leakage_rate,
        "nn_dist_syn_mean": float(np.mean(dist_syn)),
        "nn_dist_test_mean": float(np.mean(dist_test)),
    }

def privacy_membership_inference_auc(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    syn_df: pd.DataFrame,
    target: str,
    max_n: int = 5000,
    random_state: int = 66,
):
    rt = real_train.sample(min(len(real_train), max_n), random_state=random_state)
    rtest = real_test.sample(min(len(real_test), max_n), random_state=random_state)
    syn = syn_df.sample(min(len(syn_df), max_n), random_state=random_state)

    X_rt = rt.drop(columns=[target], errors="ignore")
    X_rtest = rtest.drop(columns=[target], errors="ignore")
    X_syn = syn.drop(columns=[target], errors="ignore")

    pre = build_preprocessor(X_rt)
    X_rt_enc = pre.fit_transform(X_rt)
    X_rtest_enc = pre.transform(X_rtest)
    X_syn_enc = pre.transform(X_syn)

    nn_syn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn_syn.fit(X_syn_enc)

    dist_train, _ = nn_syn.kneighbors(X_rt_enc)
    dist_test, _ = nn_syn.kneighbors(X_rtest_enc)

    dist_train = dist_train.ravel()
    dist_test = dist_test.ravel()

    y = np.concatenate([np.ones_like(dist_train), np.zeros_like(dist_test)])
    scores = -np.concatenate([dist_train, dist_test])  # closer => more "member-like"

    try:
        auc = float(roc_auc_score(y, scores))
    except Exception:
        auc = float("nan")

    return {
        "mia_auc": auc,
        "dist_train_mean": float(np.mean(dist_train)),
        "dist_test_mean": float(np.mean(dist_test)),
    }

def run_privacy_suite(real_train_s, real_test_s, syn_s, target: str, job: dict):
    max_n = int(job.get("privacy_max_n", 5000))
    percentile = float(job.get("privacy_percentile", 1.0))
    random_state = int(job.get("random_state", 66))

    cols_for_dups = [c for c in real_train_s.columns if c != target]
    dup_rate = exact_duplicate_rate(real_train_s, syn_s, cols=cols_for_dups, max_rows=max_n, random_state=random_state)
    nn = privacy_nn_leakage(real_train_s, real_test_s, syn_s, target, max_n=max_n, percentile=percentile, random_state=random_state)
    mia = privacy_membership_inference_auc(real_train_s, real_test_s, syn_s, target, max_n=max_n, random_state=random_state)

    return {"exact_duplicate_rate": float(dup_rate), **nn, **mia}


def compute_table_metric_safe(metric_cls, real_df, syn_df, meta: dict) -> float:
    try:
        return float(metric_cls.compute(real_data=real_df, synthetic_data=syn_df, metadata=meta))
    except TypeError:
        try:
            return float(metric_cls.compute(real_df, syn_df, meta))
        except Exception:
            return np.nan
    except Exception:
        return np.nan

def run_pipeline(job: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # subdirs
    (out_dir / "synthetic").mkdir(exist_ok=True)
    (out_dir / "reports" / "sdmetrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports" / "details").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports" / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "fairness").mkdir(exist_ok=True)
    (out_dir / "utility").mkdir(exist_ok=True)
    (out_dir / "privacy").mkdir(exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

    real_train, real_test = load_diabetes(job)

    meta = load_or_build_single_table_metadata(real_train, META_PATH)

    real_train_s = sanitize_for_sdmetrics(real_train, meta)
    real_test_s  = sanitize_for_sdmetrics(real_test, meta)

    baseline_pred = train_predict_lr(real_train_s, real_test_s)
    baseline_util = {k: baseline_pred[k] for k in ["auc", "f1", "acc"]}
    baseline_fair = fairness_report(baseline_pred["y_true"], baseline_pred["y_pred"], real_test_s)

    generators = job.get("generators", ["gaussian_copula", "ctgan", "tvae"])
    save_plots = bool(job.get("save_plots", True))
    subsample_rows = int(job.get("pair_metric_rows", 5000))

    num_rows = int(job.get("num_rows", len(real_train_s)))

    rows = []
    full = {
        "run_id": job["run_id"],
        "num_rows": num_rows,
        "subset_rows": int(job.get("subset_rows", 0)) or len(real_train),
        "baseline": {
            "utility_real_to_real": baseline_util,
            "fairness_real_to_real": {"dp_max": baseline_fair["dp_max"]}
        },
        "models": {}
    }

    for g in generators:
        if g not in PRETRAINED:
            continue

        model_path = PRETRAINED[g]
        model_name = {"gaussian_copula": "GaussianCopula", "ctgan": "CTGAN", "tvae": "TVAE"}.get(g, g)

        t0 = time.time()

        try:
            model = load_model(model_path)
        except Exception as e:
            load_err = f"Failed to load model {model_name}: {e}"
            rows.append({"generator": model_name, "error": load_err})
            full["models"][model_name] = {"error": load_err}
            continue

        try:
            shutil.copy2(model_path, out_dir / "models" / model_path.name)
        except Exception:
            pass

        syn_raw = model.sample(num_rows=num_rows)

        syn_aligned = align_like_real(real_train_s, syn_raw)
        syn_s = sanitize_for_sdmetrics(syn_aligned, meta)

        syn_csv = out_dir / "synthetic" / f"{model_name}_sample.csv"
        syn_s.head(5000).to_csv(syn_csv, index=False)

        diag, qual = run_reports_for_one(model_name, real_train_s, syn_s, meta, verbose=False)

        diag_pkl = out_dir / "reports" / "sdmetrics" / f"{model_name}_diagnostic_report.pkl"
        qual_pkl = out_dir / "reports" / "sdmetrics" / f"{model_name}_quality_report.pkl"
        diag.save(str(diag_pkl))
        qual.save(str(qual_pkl))

        diag_score = float(diag.get_score())
        qual_score = float(qual.get_score())

        diag_props = props_to_dict(diag.get_properties())
        qual_props = props_to_dict(qual.get_properties())

        try:
            diag_valid = diag.get_details("Data Validity")
            diag_valid.to_csv(out_dir / "reports" / "details" / f"{model_name}_diag_data_validity.csv", index=False)
        except Exception:
            pass

        try:
            qual_shapes = qual.get_details("Column Shapes")
            qual_shapes.to_csv(out_dir / "reports" / "details" / f"{model_name}_qual_column_shapes.csv", index=False)
        except Exception:
            pass

        try:
            qual_pairs = qual.get_details("Column Pair Trends")
            qual_pairs.to_csv(out_dir / "reports" / "details" / f"{model_name}_qual_column_pair_trends.csv", index=False)
        except Exception:
            pass

        if save_plots:
            try:
                diag_fig = diag.get_visualization("Data Validity")
                diag_fig.write_html(out_dir / "reports" / "plots" / f"{model_name}_diag_data_validity.html", include_plotlyjs="cdn")
            except Exception:
                pass
            try:
                shapes_fig = qual.get_visualization("Column Shapes")
                shapes_fig.write_html(out_dir / "reports" / "plots" / f"{model_name}_qual_column_shapes.html", include_plotlyjs="cdn")
            except Exception:
                pass
            try:
                pairs_fig = qual.get_visualization("Column Pair Trends")
                pairs_fig.write_html(out_dir / "reports" / "plots" / f"{model_name}_qual_column_pair_trends.html", include_plotlyjs="cdn")
            except Exception:
                pass

        colm = per_column_metrics(real_train_s, syn_s, meta)
        col_csv = out_dir / "reports" / "details" / f"{model_name}_individual_column_metrics.csv"
        colm.to_csv(col_csv, index=False)

        corr_df, cont_df = per_pair_metrics(real_train_s, syn_s, meta, subsample_rows=subsample_rows, random_state=66)
        corr_csv = out_dir / "reports" / "details" / f"{model_name}_pair_correlation_similarity.csv"
        cont_csv = out_dir / "reports" / "details" / f"{model_name}_pair_contingency_similarity.csv"
        corr_df.to_csv(corr_csv, index=False)
        cont_df.to_csv(cont_csv, index=False)

        new_row = compute_table_metric_safe(NewRowSynthesis, real_train_s, syn_s, meta)
        tbl_struct = compute_table_metric_safe(TableStructure, real_train_s, syn_s, meta)

        tstr_pred = train_predict_lr(syn_s, real_test_s)
        tstr_util = {k: tstr_pred[k] for k in ["auc", "f1", "acc"]}

        atomic_write_json(out_dir / "utility" / f"{model_name}_utility.json",
                          {"baseline" : baseline_util, "tstr" : tstr_util})

        fair = fairness_report(tstr_pred["y_true"], tstr_pred["y_pred"], real_test_s)

        if fair["race"] is not None:
            fair["race"]["by_group"].to_csv(out_dir / "fairness" / f"{model_name}_bygroup_race.csv", index=False)
        if fair["gender"] is not None:
            fair["gender"]["by_group"].to_csv(out_dir / "fairness" / f"{model_name}_bygroup_gender.csv", index=False)
        if fair["intersection"] is not None:
            fair["intersection"]["by_group"].to_csv(out_dir / "fairness" / f"{model_name}_bygroup_intersection.csv", index=False)

        dp_amplification = float(fair["dp_max"] - baseline_fair["dp_max"]) if np.isfinite(fair["dp_max"]) else np.nan

        privacy = run_privacy_suite(real_train_s, real_test_s, syn_s, TARGET, job)
        privacy_path = out_dir / "privacy" / f"{model_name}_privacy.json"
        atomic_write_json(privacy_path, privacy)      

        if save_plots:
            try:
                fig = get_column_plot(real_train_s, syn_s, column_name="race")
                fig.write_html(out_dir / "reports" / "plots" / f"{model_name}_column_race.html", include_plotlyjs="cdn")
            except Exception:
                pass
            try:
                fig = get_column_plot(real_train_s, syn_s, column_name="gender")
                fig.write_html(out_dir / "reports" / "plots" / f"{model_name}_column_gender.html", include_plotlyjs="cdn")
            except Exception:
                pass
            if "time_in_hospital" in real_train_s.columns and "num_medications" in real_train_s.columns:
                try:
                    fig = get_column_pair_plot(real_train_s, syn_s, column_names=["time_in_hospital", "num_medications"])
                    fig.write_html(out_dir / "reports" / "plots" / f"{model_name}_pair_time_vs_meds.html", include_plotlyjs="cdn")
                except Exception:
                    pass

        train_seconds = float(time.time() - t0)

        row = {
            "run_id": job["run_id"],
            "generator": model_name,
            "num_rows": num_rows,
            "subset_rows": int(job.get("subset_rows", 0)) or len(real_train_s),
            "train_seconds": train_seconds,

            "diagnostic_score": diag_score,
            "quality_score": qual_score,
            "diag_data_validity": float(diag_props.get("Data Validity", np.nan)),
            "diag_data_structure": float(diag_props.get("Data Structure", np.nan)),
            "qual_column_shapes": float(qual_props.get("Column Shapes", np.nan)),
            "qual_column_pair_trends": float(qual_props.get("Column Pair Trends", np.nan)),

            "new_row_synthesis": new_row,
            "table_structure": tbl_struct,

            "baseline_auc": baseline_util["auc"],
            "tstr_auc": tstr_util["auc"],
            "tstr_f1": tstr_util["f1"],
            "tstr_acc": tstr_util["acc"],

            "dp_max": fair["dp_max"],
            "baseline_dp_max": baseline_fair["dp_max"],
            "dp_amplification": dp_amplification,

            "privacy_exact_dup_rate": privacy["exact_duplicate_rate"],
            "privacy_nn_leakage_rate": privacy["nn_leakage_rate"],
            "privacy_nn_tau": privacy["nn_tau"],
            "privacy_mia_auc": privacy["mia_auc"],

            "synthetic_sample_csv": str(syn_csv),
            "diagnostic_report_pkl": str(diag_pkl),
            "quality_report_pkl": str(qual_pkl),
            "individual_column_metrics_csv": str(col_csv),
            "pair_corr_csv": str(corr_csv),
            "pair_contingency_csv": str(cont_csv),
            "privacy_json": str(privacy_path),
        }
        rows.append(row)

        full["models"][model_name] = {
            "reports": {"diagnostic_score": diag_score, "quality_score": qual_score},
            "utility": {"baseline": baseline_util, "tstr": tstr_util},
            "fairness": {
                "dp_max": fair["dp_max"],
                "dp_amplification": dp_amplification,
                "race": {k: fair["race"][k] for k in fair["race"] if k != "by_group"} if fair["race"] else None,
                "gender": {k: fair["gender"][k] for k in fair["gender"] if k != "by_group"} if fair["gender"] else None,
                "intersection": {k: fair["intersection"][k] for k in fair["intersection"] if k != "by_group"} if fair["intersection"] else None,
            },
            "privacy": privacy,
            "single_table_metrics": {"new_row_synthesis": new_row, "table_structure": tbl_struct},
            "artifacts": {k: row[k] for k in row if k.endswith("_csv") or k.endswith("_pkl") or k.endswith("_json")},
        }

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    payload = {
        "run_id": job["run_id"],
        "rows": rows,
        "full": full,
    }
    atomic_write_json(out_dir / "metrics.json", payload)

def main():
    print("Worker started. Waiting for jobs...")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        _, payload = r.blpop("jobs")
        job = json.loads(payload)
        run_id = job["run_id"]
        out_dir = ARTIFACT_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            set_status(run_id, "running", started_at=time.time())
            run_pipeline(job, out_dir)
            set_status(run_id, "done", finished_at=time.time())
            print(f"[DONE] {run_id}")

        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            (out_dir / "error.txt").write_text(err)
            set_status(run_id, "failed", error=str(e), finished_at=time.time())
            print(f"[FAILED] {run_id}: {e}")

if __name__ == "__main__":
    main()
