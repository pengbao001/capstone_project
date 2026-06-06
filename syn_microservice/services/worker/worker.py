import json
import os
import pickle
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_diabetes_hospital
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from redis import Redis
from sdmetrics.column_pairs import ContingencySimilarity, CorrelationSimilarity
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
from sdmetrics.single_column import (
    BoundaryAdherence,
    CategoryAdherence,
    CategoryCoverage,
    KSComplement,
    MissingValueSimilarity,
    RangeCoverage,
    StatisticSimilarity,
    TVComplement,
)
from sdmetrics.visualization import get_column_pair_plot, get_column_plot
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "/artifacts"))
TARGET = "readmit_binary"
RANDOM_STATE = 66

PRETRAINED = {
    "gaussian_copula": ARTIFACT_DIR / "gaussian_copula_diabetes.pkl",
    "ctgan": ARTIFACT_DIR / "ctgan_diabetes.pkl",
    "tvae": ARTIFACT_DIR / "tvae_diabetes.pkl",
}
DISPLAY_NAMES = {
    "gaussian_copula": "GaussianCopula",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
}
META_PATH = ARTIFACT_DIR / "diabetes_metadata.json"
DATA_PATH = ARTIFACT_DIR / "fairlearn_diabetes_hospital.pkl"

redis = Redis.from_url(REDIS_URL, decode_responses=True)


def run_key(run_id: str) -> str:
    return f"run:{run_id}"


def set_status(run_id: str, status: str, **extra):
    payload = {"status": status, **{key: str(value) for key, value in extra.items()}}
    redis.hset(run_key(run_id), mapping=payload)


def clean(value):
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, dict):
        return {key: clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return clean(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return clean(value.tolist())
    return value


def save_json(path: Path, payload):
    path.write_text(json.dumps(clean(payload), indent=2))


def relative(base: Path, path: Path) -> str:
    return str(path.relative_to(base))


def guess_sdtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def load_metadata(df: pd.DataFrame) -> dict:
    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text())
        if "tables" in metadata:
            metadata = metadata["tables"][next(iter(metadata["tables"]))]
        if "columns" in metadata:
            if "race" in metadata["columns"]:
                metadata["columns"]["race"]["sdtype"] = "categorical"
            if "gender" in metadata["columns"]:
                metadata["columns"]["gender"]["sdtype"] = "categorical"
            if TARGET in metadata["columns"]:
                metadata["columns"][TARGET]["sdtype"] = "boolean"
            return metadata

    metadata = {"columns": {}}
    for column in df.columns:
        metadata["columns"][column] = {"sdtype": guess_sdtype(df[column])}
    return metadata


def sanitize(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    out = df.copy()
    for column, info in metadata["columns"].items():
        if column not in out.columns:
            continue
        sdtype = info["sdtype"]
        if sdtype == "boolean":
            out[column] = out[column].map({0: False, 1: True, "0": False, "1": True, False: False, True: True})
            out[column] = out[column].astype("boolean")
        elif sdtype in {"categorical", "id", "other"}:
            out[column] = out[column].astype("object")
        elif sdtype == "numerical":
            out[column] = pd.to_numeric(out[column], errors="coerce")
        elif sdtype == "datetime":
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def align_columns(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> pd.DataFrame:
    syn_df = syn_df.copy()
    for column in real_df.columns:
        if column not in syn_df.columns:
            syn_df[column] = np.nan
    extra = [column for column in syn_df.columns if column not in real_df.columns]
    if extra:
        syn_df = syn_df.drop(columns=extra)
    return syn_df[real_df.columns]


def load_diabetes(job: dict):
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Local dataset file not found: {DATA_PATH}. "
        )

    data = pd.read_pickle(DATA_PATH).copy()

    if TARGET not in data.columns:
        raise ValueError(
            f"Local dataset is missing required target column '{TARGET}'."
        )

    real_train, real_test = train_test_split(
        data,
        test_size=float(job.get("test_size", 0.2)),
        random_state=int(job.get("random_state", RANDOM_STATE)),
        stratify=data[TARGET],
    )

    real_train = real_train.reset_index(drop=True)
    real_test = real_test.reset_index(drop=True)
    real_train[TARGET] = real_train[TARGET].astype(bool)
    real_test[TARGET] = real_test[TARGET].astype(bool)

    subset_rows = int(job.get("subset_rows", 0))
    if subset_rows and subset_rows < len(real_train):
        real_train = real_train.groupby(TARGET, group_keys=False).apply(
            lambda frame: frame.sample(
                max(1, int(round(len(frame) * subset_rows / len(real_train)))),
                random_state=RANDOM_STATE,
            )
        ).reset_index(drop=True)

    return real_train, real_test


def load_model(model_key: str):
    with PRETRAINED[model_key].open("rb") as f:
        return pickle.load(f)


def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category", "bool", "boolean"]).columns.tolist()
    num_cols = [col for col in X.columns if col not in cat_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def train_predict_lr(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(int)
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int)

    pre = build_preprocessor(X_train)
    clf = LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "acc": float(accuracy_score(y_test, pred)),
        "y_true": y_test,
        "y_pred": pred,
    }


def fairness_view(y_true, y_pred, sensitive_features, view_name: str) -> dict:
    frame = MetricFrame(
        metrics={
            "count": count,
            "selection_rate": selection_rate,
            "TPR": true_positive_rate,
            "FPR": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    return {
        "view": view_name,
        "dp_diff": float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)),
        "by_group": frame.by_group.reset_index(),
    }


def fairness_report(y_true, y_pred, real_test: pd.DataFrame) -> dict:
    race = fairness_view(y_true, y_pred, real_test["race"], "race")
    gender = fairness_view(y_true, y_pred, real_test["gender"], "gender")
    inter = fairness_view(y_true, y_pred, real_test["race"].astype(str) + "|" + real_test["gender"].astype(str), "intersection")
    dp_max = float(max(race["dp_diff"], gender["dp_diff"], inter["dp_diff"]))
    return {"dp_max": dp_max, "race": race, "gender": gender, "intersection": inter}


def row_hashes(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df.astype(str), index=False)


def exact_duplicate_rate(real_train: pd.DataFrame, syn_df: pd.DataFrame, max_n: int = 5000) -> float:
    real = real_train.sample(min(len(real_train), max_n), random_state=RANDOM_STATE).drop(columns=[TARGET], errors="ignore")
    syn = syn_df.sample(min(len(syn_df), max_n), random_state=RANDOM_STATE).drop(columns=[TARGET], errors="ignore")
    real_set = set(row_hashes(real).tolist())
    overlap = sum(hash_value in real_set for hash_value in row_hashes(syn).tolist())
    return float(overlap / len(syn))


def privacy_nn_leakage(real_train: pd.DataFrame, real_test: pd.DataFrame, syn_df: pd.DataFrame, max_n: int, percentile: float) -> dict:
    real_train = real_train.sample(min(len(real_train), max_n), random_state=RANDOM_STATE)
    real_test = real_test.sample(min(len(real_test), max_n), random_state=RANDOM_STATE)
    syn_df = syn_df.sample(min(len(syn_df), max_n), random_state=RANDOM_STATE)

    X_train = real_train.drop(columns=[TARGET], errors="ignore")
    X_test = real_test.drop(columns=[TARGET], errors="ignore")
    X_syn = syn_df.drop(columns=[TARGET], errors="ignore")

    pre = build_preprocessor(X_train)
    X_train = pre.fit_transform(X_train)
    X_test = pre.transform(X_test)
    X_syn = pre.transform(X_syn)

    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn.fit(X_train)
    dist_test, _ = nn.kneighbors(X_test)
    dist_syn, _ = nn.kneighbors(X_syn)
    dist_test = dist_test.ravel()
    dist_syn = dist_syn.ravel()

    tau = float(np.percentile(dist_test, percentile))
    return {
        "nn_tau": tau,
        "nn_leakage_rate": float(np.mean(dist_syn <= tau)),
        "nn_dist_syn_mean": float(np.mean(dist_syn)),
        "nn_dist_test_mean": float(np.mean(dist_test)),
    }


def privacy_membership_auc(real_train: pd.DataFrame, real_test: pd.DataFrame, syn_df: pd.DataFrame, max_n: int) -> dict:
    real_train = real_train.sample(min(len(real_train), max_n), random_state=RANDOM_STATE)
    real_test = real_test.sample(min(len(real_test), max_n), random_state=RANDOM_STATE)
    syn_df = syn_df.sample(min(len(syn_df), max_n), random_state=RANDOM_STATE)

    X_train = real_train.drop(columns=[TARGET], errors="ignore")
    X_test = real_test.drop(columns=[TARGET], errors="ignore")
    X_syn = syn_df.drop(columns=[TARGET], errors="ignore")

    pre = build_preprocessor(X_train)
    X_train = pre.fit_transform(X_train)
    X_test = pre.transform(X_test)
    X_syn = pre.transform(X_syn)

    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn.fit(X_syn)
    dist_train, _ = nn.kneighbors(X_train)
    dist_test, _ = nn.kneighbors(X_test)
    y = np.concatenate([np.ones(len(dist_train)), np.zeros(len(dist_test))])
    scores = -np.concatenate([dist_train.ravel(), dist_test.ravel()])

    return {
        "mia_auc": float(roc_auc_score(y, scores)),
        "dist_train_mean": float(np.mean(dist_train)),
        "dist_test_mean": float(np.mean(dist_test)),
    }


def privacy_report(real_train: pd.DataFrame, real_test: pd.DataFrame, syn_df: pd.DataFrame, job: dict) -> dict:
    max_n = int(job.get("privacy_max_n", 5000))
    percentile = float(job.get("privacy_percentile", 1.0))
    report = {
        "exact_duplicate_rate": exact_duplicate_rate(real_train, syn_df, max_n=max_n),
    }
    report.update(privacy_nn_leakage(real_train, real_test, syn_df, max_n=max_n, percentile=percentile))
    report.update(privacy_membership_auc(real_train, real_test, syn_df, max_n=max_n))
    return report


def run_reports(real_df: pd.DataFrame, syn_df: pd.DataFrame, metadata: dict):
    diagnostic = DiagnosticReport()
    diagnostic.generate(real_df, syn_df, metadata, verbose=False)
    quality = QualityReport()
    quality.generate(real_df, syn_df, metadata, verbose=False)
    return diagnostic, quality


def per_column_metrics(real_df: pd.DataFrame, syn_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    rows = []
    for column, info in metadata["columns"].items():
        if column not in real_df.columns or column not in syn_df.columns:
            continue
        real_col = real_df[column]
        syn_col = syn_df[column]
        sdtype = info["sdtype"]
        rows.append({"column": column, "sdtype": sdtype, "metric": "MissingValueSimilarity", "score": float(MissingValueSimilarity.compute(real_col, syn_col))})
        if sdtype in {"numerical", "datetime"}:
            rows.extend([
                {"column": column, "sdtype": sdtype, "metric": "KSComplement", "score": float(KSComplement.compute(real_col, syn_col))},
                {"column": column, "sdtype": sdtype, "metric": "RangeCoverage", "score": float(RangeCoverage.compute(real_col, syn_col))},
                {"column": column, "sdtype": sdtype, "metric": "BoundaryAdherence", "score": float(BoundaryAdherence.compute(real_col, syn_col))},
                {"column": column, "sdtype": sdtype, "metric": "StatisticSimilarity(mean)", "score": float(StatisticSimilarity.compute(real_col, syn_col, statistic="mean"))},
                {"column": column, "sdtype": sdtype, "metric": "StatisticSimilarity(median)", "score": float(StatisticSimilarity.compute(real_col, syn_col, statistic="median"))},
            ])
        else:
            rows.extend([
                {"column": column, "sdtype": sdtype, "metric": "TVComplement", "score": float(TVComplement.compute(real_col, syn_col))},
                {"column": column, "sdtype": sdtype, "metric": "CategoryCoverage", "score": float(CategoryCoverage.compute(real_col, syn_col))},
                {"column": column, "sdtype": sdtype, "metric": "CategoryAdherence", "score": float(CategoryAdherence.compute(real_col, syn_col))},
            ])
    return pd.DataFrame(rows)


def infer_column_types(metadata: dict):
    numerical = []
    categorical = []
    for column, info in metadata["columns"].items():
        if info["sdtype"] in {"numerical", "datetime"}:
            numerical.append(column)
        else:
            categorical.append(column)
    return numerical, categorical


def per_pair_metrics(real_df: pd.DataFrame, syn_df: pd.DataFrame, metadata: dict, max_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    numerical, categorical = infer_column_types(metadata)
    real_df = real_df.sample(min(len(real_df), max_rows), random_state=RANDOM_STATE)
    syn_df = syn_df.sample(min(len(syn_df), max_rows), random_state=RANDOM_STATE)

    corr_rows = [
        {
            "col_a": a,
            "col_b": b,
            "metric": "CorrelationSimilarity",
            "score": float(CorrelationSimilarity.compute(real_df[[a, b]], syn_df[[a, b]])),
        }
        for a, b in combinations(numerical, 2)
    ]
    cont_rows = [
        {
            "col_a": a,
            "col_b": b,
            "metric": "ContingencySimilarity",
            "score": float(ContingencySimilarity.compute(real_df[[a, b]], syn_df[[a, b]])),
        }
        for a, b in combinations(categorical, 2)
    ]

    corr_df = pd.DataFrame(corr_rows)
    cont_df = pd.DataFrame(cont_rows)
    if not corr_df.empty:
        corr_df = corr_df.sort_values("score")
    if not cont_df.empty:
        cont_df = cont_df.sort_values("score")
    return corr_df, cont_df


def save_group_csvs(base_dir: Path, model_name: str, fairness: dict):
    fairness["race"]["by_group"].to_csv(base_dir / "fairness" / f"{model_name}_bygroup_race.csv", index=False)
    fairness["gender"]["by_group"].to_csv(base_dir / "fairness" / f"{model_name}_bygroup_gender.csv", index=False)
    fairness["intersection"]["by_group"].to_csv(base_dir / "fairness" / f"{model_name}_bygroup_intersection.csv", index=False)


def save_plots(base_dir: Path, model_name: str, real_df: pd.DataFrame, syn_df: pd.DataFrame, diagnostic: DiagnosticReport, quality: QualityReport):
    (base_dir / "reports" / "plots").mkdir(parents=True, exist_ok=True)
    diagnostic.get_visualization("Data Validity").write_html(base_dir / "reports" / "plots" / f"{model_name}_diag_data_validity.html", include_plotlyjs="cdn")
    quality.get_visualization("Column Shapes").write_html(base_dir / "reports" / "plots" / f"{model_name}_qual_column_shapes.html", include_plotlyjs="cdn")
    quality.get_visualization("Column Pair Trends").write_html(base_dir / "reports" / "plots" / f"{model_name}_qual_column_pair_trends.html", include_plotlyjs="cdn")
    get_column_plot(real_df, syn_df, column_name="race").write_html(base_dir / "reports" / "plots" / f"{model_name}_column_race.html", include_plotlyjs="cdn")
    get_column_plot(real_df, syn_df, column_name="gender").write_html(base_dir / "reports" / "plots" / f"{model_name}_column_gender.html", include_plotlyjs="cdn")
    get_column_pair_plot(real_df, syn_df, column_names=["time_in_hospital", "num_medications"]).write_html(base_dir / "reports" / "plots" / f"{model_name}_pair_time_vs_meds.html", include_plotlyjs="cdn")


def run_pipeline(job: dict, out_dir: Path):
    for subdir in ["synthetic", "reports/details", "reports/plots", "fairness", "privacy"]:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    generators = [model_key for model_key in job.get("generators", ["gaussian_copula", "ctgan", "tvae"]) if model_key in DISPLAY_NAMES]
    total_models = len(generators)
    run_id = job["run_id"]

    set_status(run_id, "running", stage="loading_data", models_total=total_models, models_done=0, progress=0.0, current_model="")
    real_train, real_test = load_diabetes(job)
    metadata = load_metadata(real_train)
    real_train = sanitize(real_train, metadata)
    real_test = sanitize(real_test, metadata)

    real_reference_csv = out_dir / "real_reference.csv"
    real_train.to_csv(real_reference_csv, index=False)

    set_status(run_id, "running", stage="building_baseline", models_total=total_models, models_done=0, progress=0.0, current_model="")
    baseline_pred = train_predict_lr(real_train, real_test)
    baseline_fair = fairness_report(baseline_pred["y_true"], baseline_pred["y_pred"], real_test)
    baseline = {
        "utility_real_to_real": {key: baseline_pred[key] for key in ["auc", "f1", "acc"]},
        "fairness_real_to_real": {"dp_max": baseline_fair["dp_max"]},
    }

    rows = []
    full = {"run_id": run_id, "dataset": job.get("dataset", "fairlearn_diabetes_hospital"), "goal": job.get("goal", "privacy"), "baseline": baseline, "models": {}}

    for index, model_key in enumerate(generators):
        model_name = DISPLAY_NAMES[model_key]
        start = time.time()
        base_progress = index / total_models if total_models else 1.0

        set_status(run_id, "running", stage="loading_model", models_total=total_models, models_done=index, progress=base_progress, current_model=model_name)
        model = load_model(model_key)

        set_status(run_id, "running", stage="sampling_synthetic_rows", models_total=total_models, models_done=index, progress=base_progress, current_model=model_name)
        syn_raw = model.sample(num_rows=int(job.get("num_rows", len(real_train))))
        syn_df = sanitize(align_columns(real_train, syn_raw), metadata)

        syn_csv = out_dir / "synthetic" / f"{model_name}_sample.csv"
        syn_df.to_csv(syn_csv, index=False)

        set_status(run_id, "running", stage="evaluating_model", models_total=total_models, models_done=index, progress=base_progress, current_model=model_name)
        diagnostic, quality = run_reports(real_train, syn_df, metadata)
        diagnostic.get_details("Data Validity").to_csv(out_dir / "reports" / "details" / f"{model_name}_diag_data_validity.csv", index=False)
        quality.get_details("Column Shapes").to_csv(out_dir / "reports" / "details" / f"{model_name}_qual_column_shapes.csv", index=False)
        quality.get_details("Column Pair Trends").to_csv(out_dir / "reports" / "details" / f"{model_name}_qual_column_pair_trends.csv", index=False)

        if job.get("save_plots", True):
            save_plots(out_dir, model_name, real_train, syn_df, diagnostic, quality)

        column_metrics = per_column_metrics(real_train, syn_df, metadata)
        pair_corr, pair_cont = per_pair_metrics(real_train, syn_df, metadata, max_rows=int(job.get("pair_metric_rows", 5000)))
        column_csv = out_dir / "reports" / "details" / f"{model_name}_individual_column_metrics.csv"
        pair_corr_csv = out_dir / "reports" / "details" / f"{model_name}_pair_correlation_similarity.csv"
        pair_cont_csv = out_dir / "reports" / "details" / f"{model_name}_pair_contingency_similarity.csv"
        column_metrics.to_csv(column_csv, index=False)
        pair_corr.to_csv(pair_corr_csv, index=False)
        pair_cont.to_csv(pair_cont_csv, index=False)

        tstr = train_predict_lr(syn_df, real_test)
        fairness = fairness_report(tstr["y_true"], tstr["y_pred"], real_test)
        save_group_csvs(out_dir, model_name, fairness)

        privacy = privacy_report(real_train, real_test, syn_df, job)
        privacy_path = out_dir / "privacy" / f"{model_name}_privacy.json"
        save_json(privacy_path, privacy)

        row = {
            "generator": model_name,
            "train_seconds": float(time.time() - start),
            "diagnostic_score": float(diagnostic.get_score()),
            "quality_score": float(quality.get_score()),
            "tstr_auc": tstr["auc"],
            "tstr_f1": tstr["f1"],
            "tstr_acc": tstr["acc"],
            "dp_max": fairness["dp_max"],
            "baseline_dp_max": baseline_fair["dp_max"],
            "privacy_exact_dup_rate": privacy["exact_duplicate_rate"],
            "privacy_nn_leakage_rate": privacy["nn_leakage_rate"],
            "privacy_mia_auc": privacy["mia_auc"],
            "real_reference_csv": relative(out_dir, real_reference_csv),
            "synthetic_sample_csv": relative(out_dir, syn_csv),
            "individual_column_metrics_csv": relative(out_dir, column_csv),
            "pair_corr_csv": relative(out_dir, pair_corr_csv),
            "pair_contingency_csv": relative(out_dir, pair_cont_csv),
            "privacy_json": relative(out_dir, privacy_path),
        }
        rows.append(row)

        full["models"][model_name] = {
            "generator": model_name,
            "reports": {
                "diagnostic_score": row["diagnostic_score"],
                "quality_score": row["quality_score"],
            },
            "utility": {
                "baseline": baseline["utility_real_to_real"],
                "tstr": {key: tstr[key] for key in ["auc", "f1", "acc"]},
            },
            "fairness": {
                "dp_max": fairness["dp_max"],
                "race": {"dp_diff": fairness["race"]["dp_diff"]},
                "gender": {"dp_diff": fairness["gender"]["dp_diff"]},
                "intersection": {"dp_diff": fairness["intersection"]["dp_diff"]},
            },
            "privacy": privacy,
            "artifacts": {key: value for key, value in row.items() if key.endswith(("_csv", "_json"))},
        }

        set_status(
            run_id,
            "running",
            stage="model_complete",
            models_total=total_models,
            models_done=index + 1,
            progress=(index + 1) / total_models if total_models else 1.0,
            current_model=model_name,
        )

    set_status(run_id, "running", stage="saving_results", models_total=total_models, models_done=total_models, progress=1.0, current_model="")
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
    save_json(out_dir / "metrics.json", {"run_id": run_id, "rows": rows, "full": full})


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print("worker started")
    while True:
        _, payload = redis.blpop("jobs")
        job = json.loads(payload)
        run_id = job["run_id"]
        out_dir = ARTIFACT_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            set_status(run_id, "running", started_at=time.time(), stage="starting", current_model="", models_done=0, progress=0.0)
            run_pipeline(job, out_dir)
            total_models = len([model_key for model_key in job.get("generators", ["gaussian_copula", "ctgan", "tvae"]) if model_key in DISPLAY_NAMES])
            set_status(run_id, "done", finished_at=time.time(), stage="done", current_model="", models_done=total_models, models_total=total_models, progress=1.0)
        except Exception as exc:
            (out_dir / "error.txt").write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            set_status(run_id, "failed", error=str(exc), finished_at=time.time(), stage="failed")


if __name__ == "__main__":
    main()
