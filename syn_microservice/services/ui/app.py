import os
import re
import time
from io import StringIO
from pathlib import Path, PurePosixPath

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

API_URL = os.environ.get("API_URL", "http://api:8000")

st.set_page_config(page_title="Synthetic Health Dashboard", layout="wide")
st.title("Synthetic Health Data Dashboard (SDV + Fairness + Privacy)")


def safe_json(resp: requests.Response):
    try:
        return resp.json(), None
    except Exception as e:
        text = resp.text or ""
        snippet = text[:500].replace("\n", "\\n")
        return None, (
            f"JSON parse failed: {e}. status={resp.status_code}, "
            f"content-type={resp.headers.get('content-type')}, body[:500]={snippet}"
        )


def get_json(url: str, timeout: int = 10):
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        return None, None, f"Request failed: {e}"

    if resp.status_code != 200:
        text = resp.text or ""
        snippet = text[:500].replace("\n", "\\n")
        return None, resp, (
            f"HTTP {resp.status_code}. "
            f"content-type={resp.headers.get('content-type')}, body[:500]={snippet}"
        )

    data, err = safe_json(resp)
    return data, resp, err


def _normalize_artifact_path(run_id: str, artifact_path: str) -> tuple[str, str]:
    parts = PurePosixPath(artifact_path).parts
    if "artifacts" in parts:
        idx = parts.index("artifacts")
        if len(parts) >= idx + 3:
            return parts[idx + 1], "/".join(parts[idx + 2 :])
    return run_id, artifact_path


def load_artifact_csv(run_id: str, artifact_path: str) -> pd.DataFrame:
    path = Path(artifact_path)

    if path.is_absolute() and path.exists():
        return pd.read_csv(path)

    local_shared = Path("/artifacts") / run_id / artifact_path
    if local_shared.exists():
        return pd.read_csv(local_shared)

    fixed_run_id, rel_path = _normalize_artifact_path(run_id, artifact_path)
    resp = requests.get(
        f"{API_URL}/runs/{fixed_run_id}/artifact",
        params={"path": rel_path},
        timeout=30,
    )
    if resp.status_code == 200:
        return pd.read_csv(StringIO(resp.text))

    raise FileNotFoundError(f"Could not load artifact CSV: {artifact_path}")


def load_artifact_text(run_id: str, artifact_path: str) -> str:
    path = Path(artifact_path)

    if path.is_absolute() and path.exists():
        return path.read_text()

    local_shared = Path("/artifacts") / run_id / artifact_path
    if local_shared.exists():
        return local_shared.read_text()

    fixed_run_id, rel_path = _normalize_artifact_path(run_id, artifact_path)
    resp = requests.get(
        f"{API_URL}/runs/{fixed_run_id}/artifact",
        params={"path": rel_path},
        timeout=30,
    )
    if resp.status_code == 200:
        return resp.text

    raise FileNotFoundError(f"Could not load artifact text: {artifact_path}")


def sort_age_bins(values: pd.Series) -> list[str]:
    unique_values = pd.Series(values).dropna().astype(str).unique().tolist()

    def key(value: str) -> int:
        match = re.search(r"(\d+)", value)
        return int(match.group(1)) if match else 10**9

    return sorted(unique_values, key=key)


def series_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    nonnull = series.dropna()
    lowered = set(nonnull.astype(str).str.lower().unique())
    if lowered and lowered <= {"0", "1", "true", "false"}:
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    return "categorical"


def to_binary_rate(series: pd.Series) -> pd.Series:
    mapped = series.map(
        {
            True: 1,
            False: 0,
            "True": 1,
            "False": 0,
            "true": 1,
            "false": 0,
            1: 1,
            0: 0,
            "1": 1,
            "0": 0,
        }
    )
    return pd.to_numeric(mapped, errors="coerce")


def build_age_trend_table(real_df: pd.DataFrame, syn_df: pd.DataFrame, y_col: str, y_category: str | None = None):
    age_order = sort_age_bins(pd.concat([real_df["age"], syn_df["age"]], ignore_index=True))
    kind = series_kind(real_df[y_col])

    if kind == "numeric":
        real_series = pd.to_numeric(real_df[y_col], errors="coerce").groupby(real_df["age"]).mean()
        syn_series = pd.to_numeric(syn_df[y_col], errors="coerce").groupby(syn_df["age"]).mean()
        value_label = f"Mean {y_col}"
    elif kind == "boolean":
        real_series = to_binary_rate(real_df[y_col]).groupby(real_df["age"]).mean()
        syn_series = to_binary_rate(syn_df[y_col]).groupby(syn_df["age"]).mean()
        value_label = f"Rate of {y_col}=True"
    else:
        if y_category is None:
            raise ValueError("y_category is required for categorical columns")
        real_series = real_df[y_col].astype(str).eq(str(y_category)).astype(float).groupby(real_df["age"]).mean()
        syn_series = syn_df[y_col].astype(str).eq(str(y_category)).astype(float).groupby(syn_df["age"]).mean()
        value_label = f"Share of {y_col} = {y_category}"

    comp = pd.DataFrame({"age": age_order})
    comp["Real"] = comp["age"].map(real_series)
    comp["Synthetic"] = comp["age"].map(syn_series)
    comp["Gap"] = (comp["Real"] - comp["Synthetic"]).abs()
    return comp, value_label

def explain_trend_match(comp: pd.DataFrame):
    valid = comp.dropna(subset=["Real", "Synthetic"])
    if len(valid) < 2:
        return "Not enough data", "There are not enough shared age groups to compare the trend."

    corr = valid["Real"].corr(valid["Synthetic"])
    avg_gap = float(valid["Gap"].mean())
    worst = valid.sort_values("Gap", ascending=False).iloc[0]

    if pd.isna(corr):
        strength = "unclear"
    elif corr >= 0.9:
        strength = "very similar"
    elif corr >= 0.7:
        strength = "similar"
    elif corr >= 0.4:
        strength = "moderately similar"
    else:
        strength = "not very similar"

    title = f"Trend match: {strength}"
    body = (
        f"Across the age groups that appear in both datasets, the real and synthetic trends are {strength}. "
        f"The average gap is {avg_gap:.4f}. "
        f"The biggest difference appears at age group {worst['age']} with a gap of {worst['Gap']:.4f}."
    )
    return title, body


def fetch_manifest(run_id: str, model_name: str):
    data, _, err = get_json(f"{API_URL}/runs/{run_id}/models/{model_name}/trend-manifest", timeout=30)
    return data, err


def render_html_artifact(run_id: str, artifact_path: str, height: int = 650):
    try:
        html = load_artifact_text(run_id, artifact_path)
        components.html(html, height=height, scrolling=True)
    except Exception as e:
        st.error(f"Could not render artifact {artifact_path}: {e}")


def show_age_trend_story(run_id: str, model_payload: dict):
    artifacts = model_payload.get("artifacts", {})
    real_path = artifacts.get("real_reference_csv")
    syn_path = artifacts.get("synthetic_sample_csv")

    if not real_path or not syn_path:
        st.info("This run does not include the real reference CSV yet. Re-run with the current worker.")
        return

    try:
        real_df = load_artifact_csv(run_id, real_path)
        syn_df = load_artifact_csv(run_id, syn_path)
    except Exception as e:
        st.error(f"Could not load the CSV files needed for the trend story: {e}")
        return

    if "age" not in real_df.columns or "age" not in syn_df.columns:
        st.warning("This dataset does not contain an age column, so the age-based trend story cannot be shown.")
        return

    st.caption("This view compares one simple relationship at a time in the real and synthetic datasets.")
    preset = st.selectbox(
        "Choose a story",
        ["Age → Num Medications", "Age → Readmission Rate", "Age → Insulin Category", "Custom"],
        key=f"trend_story_preset_{model_payload.get('generator', 'model')}",
    )

    if preset == "Age → Num Medications":
        y_col = "num_medications"
    elif preset == "Age → Readmission Rate":
        y_col = "readmit_binary"
    elif preset == "Age → Insulin Category":
        y_col = "insulin"
    else:
        y_col = st.selectbox(
            "Y column",
            [column for column in real_df.columns if column != "age"],
            key=f"trend_story_y_{model_payload.get('generator', 'model')}",
        )

    if y_col not in real_df.columns or y_col not in syn_df.columns:
        st.warning(f"The selected column '{y_col}' is not available in both datasets.")
        return

    kind = series_kind(real_df[y_col])
    y_category = None
    if kind == "categorical":
        categories = sorted(pd.concat([real_df[y_col], syn_df[y_col]], ignore_index=True).dropna().astype(str).unique().tolist())
        if not categories:
            st.warning(f"No categories were found for '{y_col}'.")
            return
        y_category = st.selectbox(
            f"Category to track in {y_col}",
            categories,
            key=f"trend_story_category_{model_payload.get('generator', 'model')}_{y_col}",
        )

    comp, value_label = build_age_trend_table(real_df, syn_df, y_col, y_category=y_category)
    st.markdown(f"**Age → {value_label}**")
    st.line_chart(comp.set_index("age")[["Real", "Synthetic"]], height=360, use_container_width=True)
    st.markdown("**Absolute gap between the two trends**")
    st.bar_chart(comp.set_index("age")[["Gap"]], height=280, use_container_width=True)

    title, body = explain_trend_match(comp)
    st.success(title)
    st.write(body)
    with st.expander("Trend comparison table"):
        st.dataframe(comp, use_container_width=True)


def load_real_and_synthetic(run_id: str, model_payload: dict):
    artifacts = model_payload.get("artifacts", {})
    real_path = artifacts.get("real_reference_csv")
    syn_path = artifacts.get("synthetic_sample_csv")
    if not real_path or not syn_path:
        raise FileNotFoundError("real_reference_csv or synthetic_sample_csv is missing from model artifacts")
    return load_artifact_csv(run_id, real_path), load_artifact_csv(run_id, syn_path)


def combined_frame(real_df: pd.DataFrame, syn_df: pd.DataFrame, column: str) -> pd.DataFrame:
    real = pd.DataFrame({column: real_df[column], "dataset": "real"})
    syn = pd.DataFrame({column: syn_df[column], "dataset": "synthetic"})
    return pd.concat([real, syn], ignore_index=True)


def show_numeric_feature_plots(real_df: pd.DataFrame, syn_df: pd.DataFrame, column: str, key: str):
    plot_df = combined_frame(real_df, syn_df, column)
    hist = px.histogram(plot_df, x=column, color="dataset", barmode="overlay", opacity=0.65, nbins=20, title=f"{column}: distribution")
    hist.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(hist, use_container_width=True, key=f"hist_{key}_{column}")

    box = px.box(plot_df, x="dataset", y=column, color="dataset", title=f"{column}: box plot")
    box.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(box, use_container_width=True, key=f"box_{key}_{column}")


def show_categorical_feature_plots(real_df: pd.DataFrame, syn_df: pd.DataFrame, column: str, key: str):
    plot_df = combined_frame(real_df, syn_df, column)
    plot_df[column] = plot_df[column].astype(str)
    counts = plot_df.groupby([column, "dataset"]).size().reset_index(name="count")
    fig = px.bar(counts, x=column, y="count", color="dataset", barmode="group", title=f"{column}: category counts")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True, key=f"bar_{key}_{column}")


def show_relationship_plot(real_df: pd.DataFrame, syn_df: pd.DataFrame, primary: str, secondary: str, key: str):
    primary_kind = series_kind(real_df[primary])
    secondary_kind = series_kind(real_df[secondary])

    real = real_df[[primary, secondary]].copy()
    real["dataset"] = "real"
    syn = syn_df[[primary, secondary]].copy()
    syn["dataset"] = "synthetic"
    plot_df = pd.concat([real, syn], ignore_index=True)

    if primary_kind == "numeric" and secondary_kind == "numeric":
        sampled = plot_df.sample(min(len(plot_df), 2000), random_state=0)
        fig = px.scatter(sampled, x=primary, y=secondary, color="dataset", opacity=0.45, title=f"{primary} vs {secondary}")
    elif primary_kind == "numeric" and secondary_kind != "numeric":
        plot_df[secondary] = plot_df[secondary].astype(str)
        fig = px.violin(plot_df, x=secondary, y=primary, color="dataset", box=True, points=False, title=f"{primary} by {secondary}")
    elif primary_kind != "numeric" and secondary_kind == "numeric":
        plot_df[primary] = plot_df[primary].astype(str)
        fig = px.violin(plot_df, x=primary, y=secondary, color="dataset", box=True, points=False, title=f"{secondary} by {primary}")
    else:
        real_ct = pd.crosstab(real_df[primary].astype(str), real_df[secondary].astype(str), normalize="index")
        syn_ct = pd.crosstab(syn_df[primary].astype(str), syn_df[secondary].astype(str), normalize="index")
        diff = real_ct.subtract(syn_ct, fill_value=0)
        diff = diff.sort_index().sort_index(axis=1)
        fig = px.imshow(diff, aspect="auto", title=f"Difference heatmap: {primary} vs {secondary} (real - synthetic)")

    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True, key=f"rel_{key}_{primary}_{secondary}")


def show_all_feature_grid(real_df: pd.DataFrame, syn_df: pd.DataFrame, model_key: str):
    columns = list(real_df.columns)
    grid = st.columns(3)
    for idx, column in enumerate(columns):
        with grid[idx % 3]:
            kind = series_kind(real_df[column])
            st.markdown(f"**{column}**")
            if kind == "numeric":
                plot_df = combined_frame(real_df, syn_df, column)
                fig = px.histogram(plot_df, x=column, color="dataset", barmode="overlay", opacity=0.65, nbins=16)
            else:
                plot_df = combined_frame(real_df, syn_df, column)
                plot_df[column] = plot_df[column].astype(str)
                counts = plot_df.groupby([column, "dataset"]).size().reset_index(name="count")
                fig = px.bar(counts, x=column, y="count", color="dataset", barmode="group")
            fig.update_layout(height=260, showlegend=(idx == 0), margin=dict(l=8, r=8, t=10, b=8))
            st.plotly_chart(fig, use_container_width=True, key=f"all_{model_key}_{column}")


def show_visualize_tab(run_id: str, model_name: str, model_payload: dict, manifest: dict | None):
    try:
        real_df, syn_df = load_real_and_synthetic(run_id, model_payload)
    except Exception as e:
        st.error(f"Could not load real/synthetic CSVs for visualization: {e}")
        return

    if not PLOTLY_AVAILABLE:
        st.info("Install plotly in the app container to enable interactive feature plots.")
        return

    st.caption("Compare distributions and relationships between the real reference data and the synthetic sample for this model.")
    controls_left, controls_right = st.columns(2)
    with controls_left:
        primary_feature = st.selectbox(
            "Show feature",
            real_df.columns.tolist(),
            index=real_df.columns.get_loc("age") if "age" in real_df.columns else 0,
            key=f"primary_feature_{model_name}",
        )
    with controls_right:
        secondary_feature = st.selectbox(
            "Show secondary feature",
            [c for c in real_df.columns if c != primary_feature],
            index=0,
            key=f"secondary_feature_{model_name}_{primary_feature}",
        )

    show_all = st.checkbox("Show all features", value=False, key=f"show_all_features_{model_name}")

    if show_all:
        show_all_feature_grid(real_df, syn_df, model_name)
    else:
        main_left, main_right = st.columns(2)
        with main_left:
            if series_kind(real_df[primary_feature]) == "numeric":
                show_numeric_feature_plots(real_df, syn_df, primary_feature, model_name)
            else:
                show_categorical_feature_plots(real_df, syn_df, primary_feature, model_name)
        with main_right:
            show_relationship_plot(real_df, syn_df, primary_feature, secondary_feature, model_name)

    if manifest:
        st.markdown("### Saved report plots")
        with st.expander("Column Shapes (all features)", expanded=False):
            plot_path = manifest.get("plots", {}).get("qual_column_shapes_html")
            if plot_path:
                render_html_artifact(run_id, plot_path, height=720)
            else:
                st.info("This run does not include the saved Column Shapes HTML plot.")

        with st.expander("Column Pair Trends (all pairs)", expanded=False):
            plot_path = manifest.get("plots", {}).get("qual_column_pair_trends_html")
            if plot_path:
                render_html_artifact(run_id, plot_path, height=720)
            else:
                st.info("This run does not include the saved Column Pair Trends HTML plot.")

        with st.expander("Data Validity", expanded=False):
            plot_path = manifest.get("plots", {}).get("diag_data_validity_html")
            if plot_path:
                render_html_artifact(run_id, plot_path, height=650)
            else:
                st.info("This run does not include the saved Data Validity HTML plot.")


def metric_row(label: str, value, help_text: str):
    st.markdown(f"- **{label}:** `{value}`")
    with st.expander(f"{label}: explanation"):
        st.write(help_text)


def ranking_proxy(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty or "quality_score" not in rows_df or "tstr_auc" not in rows_df:
        return pd.DataFrame()
    ranking = rows_df[["generator", "quality_score", "tstr_auc", "dp_max", "privacy_nn_leakage_rate"]].copy()
    ranking["quality_rank"] = ranking["quality_score"].rank(ascending=False, method="min")
    ranking["tstr_rank"] = ranking["tstr_auc"].rank(ascending=False, method="min")
    ranking["fairness_rank"] = ranking["dp_max"].rank(ascending=True, method="min")
    ranking["privacy_rank"] = ranking["privacy_nn_leakage_rate"].rank(ascending=True, method="min")
    return ranking.sort_values(["quality_rank", "tstr_rank"]) 


def show_evaluate_tab(run_id: str, model_name: str, model_payload: dict, full_payload: dict, rows_df: pd.DataFrame, manifest: dict | None):
    reports = model_payload.get("reports", {})
    utility = model_payload.get("utility", {})
    privacy = model_payload.get("privacy", {})
    fairness = model_payload.get("fairness", {})

    st.markdown("## Evaluate")
    st.markdown("### Quality and privacy metrics")
    st.write("Quality metrics show how faithfully the synthetic data matches the original data. Privacy metrics show how easy it is to recover or detect information from the original data using the synthetic sample.")

    metric_row(
        "SDMetrics quality score",
        f"{reports.get('quality_score', 'n/a')}",
        "Higher is better. This is the overall fidelity score from the SDMetrics quality report. A value closer to 1 means the synthetic table matches the real table more closely.",
    )
    metric_row(
        "Diagnostic score",
        f"{reports.get('diagnostic_score', 'n/a')}",
        "Higher is better. This checks whether the synthetic data is structurally valid for the schema and data types.",
    )
    metric_row(
        "Exact duplicate rate",
        f"{privacy.get('exact_duplicate_rate', 'n/a')}",
        "Lower is better. This is the fraction of sampled synthetic rows that exactly match sampled real training rows.",
    )
    metric_row(
        "Nearest-neighbor leakage rate",
        f"{privacy.get('nn_leakage_rate', 'n/a')}",
        "Lower is better. This estimates how often synthetic rows fall unusually close to training rows compared with held-out real rows.",
    )
    metric_row(
        "Membership inference AUC",
        f"{privacy.get('mia_auc', 'n/a')}",
        "Closer to 0.5 is better. A value far above 0.5 suggests training rows may be easier to distinguish from held-out rows using distance to the synthetic data.",
    )

    st.markdown("### Downstream performance metrics")
    with st.expander("Downstream performance metrics: guide", expanded=True):
        st.markdown(
            """
            1. Pick a **target variable** to predict.  
            2. Use the other variables as input features.  
            3. Train a **prediction model** on synthetic data.  
            4. Test that model on held-out real data.  
            
            In the current backend, this evaluation is fixed to **`readmit_binary`** with **Logistic Regression**. That means the metrics below already reflect a Train-on-Synthetic, Test-on-Real (TS-TR / TSTR) evaluation.
            """
        )

    target_left, model_right = st.columns(2)
    with target_left:
        st.selectbox("Target variable", ["readmit_binary"], index=0, key=f"target_var_{model_name}")
    with model_right:
        st.selectbox("Prediction model", ["Logistic Regression"], index=0, key=f"pred_model_{model_name}")

    if st.button("Evaluate", key=f"evaluate_{model_name}"):
        st.session_state[f"evaluate_pressed_{model_name}"] = True

    if st.session_state.get(f"evaluate_pressed_{model_name}", False):
        baseline = utility.get("baseline", {})
        tstr = utility.get("tstr", {})
        eval_df = pd.DataFrame(
            [
                {"dataset": "real→real baseline", "auc": baseline.get("auc"), "f1": baseline.get("f1"), "accuracy": baseline.get("acc")},
                {"dataset": "synthetic→real (TSTR)", "auc": tstr.get("auc"), "f1": tstr.get("f1"), "accuracy": tstr.get("acc")},
            ]
        )
        st.dataframe(eval_df, use_container_width=True)

        with st.expander("TS-TR / TSTR explanation"):
            st.write(
                "TSTR stands for Train on Synthetic, Test on Real. It measures whether a synthetic dataset preserves enough signal for a downstream model trained on synthetic rows to still work on unseen real rows."
            )

        rank_df = ranking_proxy(rows_df)
        if not rank_df.empty:
            st.markdown("#### Ranking summary across models")
            st.dataframe(rank_df, use_container_width=True)
            with st.expander("Ranking summary explanation"):
                st.write(
                    "This is a simple ranking summary across the current models. It is not a formal SRA implementation, but it helps you see which models rank well by fidelity, downstream utility, fairness, and privacy leakage at the same time."
                )

        st.markdown("### Fairness summary")
        st.write(f"Maximum demographic parity difference for this model: **{fairness.get('dp_max', 'n/a')}**")
        if manifest:
            fair_tabs = st.tabs(["Race", "Gender", "Intersection"])
            for tab, key in zip(fair_tabs, ["race_csv", "gender_csv", "intersection_csv"]):
                with tab:
                    csv_path = manifest.get("fairness", {}).get(key)
                    if csv_path:
                        st.dataframe(load_artifact_csv(run_id, csv_path), use_container_width=True)
                    else:
                        st.info("This fairness view is not available in the saved artifacts.")


def show_synthetic_data_tab(run_id: str, model_name: str, model_payload: dict):
    try:
        real_df, syn_df = load_real_and_synthetic(run_id, model_payload)
    except Exception as e:
        st.error(f"Could not load real/synthetic CSVs: {e}")
        return

    st.markdown("### Synthetic dataset preview")
    st.caption("Use this section to inspect the synthetic sample generated by the selected model and compare it with the real reference dataset used for evaluation.")
    preview_rows = st.slider("Rows to preview", min_value=5, max_value=min(100, len(syn_df)), value=min(20, len(syn_df)), key=f"preview_rows_{model_name}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Real reference preview**")
        st.caption(f"Rows: {len(real_df)} | Columns: {len(real_df.columns)}")
        st.dataframe(real_df.head(preview_rows), use_container_width=True)
    with right:
        st.markdown(f"**Synthetic preview — {model_name}**")
        st.caption(f"Rows: {len(syn_df)} | Columns: {len(syn_df.columns)}")
        st.dataframe(syn_df.head(preview_rows), use_container_width=True)
        st.download_button(
            label=f"Download {model_name} synthetic CSV",
            data=syn_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{model_name}_synthetic_sample.csv",
            mime="text/csv",
            key=f"download_{model_name}",
        )


def overview_cards(model_payload: dict):
    reports = model_payload.get("reports", {})
    utility = model_payload.get("utility", {}).get("tstr", {})
    fairness = model_payload.get("fairness", {})
    privacy = model_payload.get("privacy", {})

    cols = st.columns(5)
    cols[0].metric("Quality", f"{reports.get('quality_score', 0):.3f}" if reports.get('quality_score') is not None else "n/a")
    cols[1].metric("Diagnostic", f"{reports.get('diagnostic_score', 0):.3f}" if reports.get('diagnostic_score') is not None else "n/a")
    cols[2].metric("TSTR AUC", f"{utility.get('auc', 0):.3f}" if utility.get('auc') is not None else "n/a")
    cols[3].metric("DP max", f"{fairness.get('dp_max', 0):.3f}" if fairness.get('dp_max') is not None else "n/a")
    cols[4].metric("MIA AUC", f"{privacy.get('mia_auc', 0):.3f}" if privacy.get('mia_auc') is not None else "n/a")


def show_result_ui(run_id: str, full: dict, rows_df: pd.DataFrame):
    model_names = list(full.get("models", {}).keys())
    if not model_names:
        st.warning("No per-model details were found in the results payload.")
        return

    chosen = st.selectbox("Select model", model_names, index=0)

    explain_key = f"llm_explanation::{run_id}::{chosen}"

    if st.button("Explain selected model", key=f"btn_explain_{chosen}"):
        try:
            resp = requests.post(
                f"{API_URL}/llm/explain",
                json={"run_id" : run_id, "model_name" : chosen},
                timeout=120,
            )
            resp.raise_for_status()
            st.session_state[explain_key] = resp.json()["text"]
        except Exception as e:
            st.error(f"Couldn't get explanation: {e}")

    if explain_key in st.session_state:
        st.markdown("### LLM Explanation")
        st.write(st.session_state[explain_key])

    model_payload = full["models"][chosen]
    manifest, manifest_err = fetch_manifest(run_id, chosen)
    if manifest_err:
        manifest = None
        st.info(f"Could not load the trend manifest for {chosen}. The basic tabs will still work. {manifest_err}")

    overview_tab, data_tab, visualize_tab, trend_tab, evaluate_tab, raw_tab = st.tabs(
        ["Overview", "Synthetic Data", "Visualize", "Trend Story", "Evaluate", "Raw JSON"]
    )

    with overview_tab:
        overview_cards(model_payload)
        st.markdown("### Model summary")
        st.json(model_payload)

    with data_tab:
        show_synthetic_data_tab(run_id, chosen, model_payload)

    with visualize_tab:
        show_visualize_tab(run_id, chosen, model_payload, manifest)

    with trend_tab:
        st.markdown("### Trend Story")
        show_age_trend_story(run_id, model_payload)

    with evaluate_tab:
        show_evaluate_tab(run_id, chosen, model_payload, full, rows_df, manifest)

    with raw_tab:
        st.json(model_payload)

DEFAULTS = {
    "dataset" : "diabetes",
    "generators" : ["gaussian_copula", "ctgan", "tvae"],
    "num_rows" : 5000,
    "subset_rows" : 10000,
    "pair_metric_rows" : 5000,
    "privacy_max_n" : 5000,
    "privacy_percentile" : 1.0,
    "save_plots" : True,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.header("LLM Copilot")

    llm_goal = st.text_area(
        "Describe what you want",
        placeholder="Example: fast privacy-focued run with all 3 generators"
    )

    if st.button("Suggest config"):
        try:
            resp = requests.post(
                f"{API_URL}/llm/plan",
                json={"user_goal" : llm_goal},
                timeout=120,
            )
            resp.raise_for_status()
            plan = resp.json()["plan"]

            st.session_state["dataset"] = plan["dataset"]
            st.session_state["generators"] = plan["generators"]
            st.session_state["num_rows"] = int(plan["num_rows"])
            st.session_state["subset_rows"] = int(plan["subset_rows"])
            st.session_state["pair_metric_rows"] = int(plan["pair_metric_rows"])
            st.session_state["privacy_max_n"] = int(plan["privacy_max_n"])
            st.session_state["privacy_percentile"] = float(plan["privacy_percentile"])
            st.session_state["save_plots"] = bool(plan["save_plots"])

            st.success("Suggested config applied.")
        except Exception as e:
            st.error(f"Couldn't get plan: {e}")

    st.divider()
    st.header("Run Config")

    st.text_input(
        "Dataset name",
        key="dataset",
    )

    st.multiselect(
        "Generators",
        ["gaussian_copula", "ctgan", "tvae"],
        key="generators",
    )

    st.number_input(
        "Synthetic rows per model",
        min_value=100,
        max_value=20000,
        step=100,
        key="num_rows",
    )

    st.subheader("Speed knobs")

    st.number_input(
        "Subset rows (real_train)",
        min_value=0,
        max_value=10000,
        step=1000,
        key="subset_rows",
        help="0 = use full training split",
    )

    st.number_input(
        "Pair metric subsample rows",
        min_value=500,
        max_value=20000,
        step=500,
        key="pair_metric_rows",
    )

    st.subheader("Privacy knobs")

    st.number_input(
        "privacy sample size",
        min_value=500,
        max_value=20000,
        step=500,
        key="privacy_max_n",
    )

    st.slider(
        "NN leakage percentile",
        min_value=0.1,
        max_value=10.0,
        step=0.1,
        key="privacy_percentile",
    )

    st.checkbox(
        "Save HTML plots into artifacts",
        key="save_plots",
    )

    st.caption(f"API_URL : {API_URL}")

    if st.button("Run Experiment", type="primary"):
        payload = {
            "dataset" : st.session_state["dataset"],
            "generators" : st.session_state["generators"],
            "num_rows" : st.session_state["num_rows"],
            "subset_rows" : st.session_state["subset_rows"],
            "pair_metric_rows" : st.session_state["pair_metric_rows"],
            "privacy_max_n" : st.session_state["privacy_max_n"],
            "privacy_percentile" : st.session_state["privacy_percentile"],
            "save_plots" : st.session_state["save_plots"],
        }

        try:
            resp = requests.post(f"{API_URL}/runs", json=payload, timeout=60)
            resp.raise_for_status()
            run = resp.json()
            st.session_state["run_id"] = run["run_id"]
            st.success(f"Started run {run['run_id']}")
        except Exception as e:
            st.error(f"Couldn't start run: {e}")


run_id = st.session_state.get("run_id")
if not run_id:
    st.info("Click **Run Experiment** to start a job.")
    st.stop()

st.subheader(f"Run: {run_id}")
status_box = st.empty()

for _ in range(180):
    info, _, err = get_json(f"{API_URL}/runs/{run_id}", timeout=10)
    if err:
        status_box.error(f"Could not read run status. {err}")
        break

    status = info.get("status", "unknown")
    status_box.write({"status": status, **{key: info[key] for key in info if key != "job_json"}})

    if status == "failed":
        st.error("Run failed. Check artifacts/<run_id>/error.txt")
        break

    if status == "done":
        try:
            res_resp = requests.get(f"{API_URL}/runs/{run_id}/results", timeout=30)
        except Exception as e:
            st.error(f"GET /results request failed: {e}")
            break

        if res_resp.status_code == 404:
            time.sleep(1)
            continue
        if res_resp.status_code != 200:
            st.error(f"GET /results failed: {res_resp.status_code} body[:500]={res_resp.text[:500]}")
            break

        results, jerr = safe_json(res_resp)
        if jerr:
            st.warning("Results returned, but JSON parsing failed. Retrying...")
            st.code(jerr)
            time.sleep(1)
            continue

        rows = results.get("rows", [])
        full = results.get("full", {})
        if rows:
            df = pd.DataFrame(rows)
            st.markdown("### Model Summary")
            st.dataframe(df, use_container_width=True)
        else:
            df = pd.DataFrame()
            st.warning("No rows in results payload.")

        st.markdown("### Baseline (Real → Real)")
        st.json(full.get("baseline", {}))
        st.markdown("### Results Explorer")
        show_result_ui(run_id, full, df)
        break

    time.sleep(1)
else:
    st.warning("Still running... check again in a moment.")