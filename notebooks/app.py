import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Fairness Baseline Report (Real vs Synthetic Training)")

report_dir = Path("../artifacts/fairness_report")
summary_path = report_dir / "fairness_summary.csv"
summary = pd.read_csv(summary_path)

st.subheader("Summary table")
st.dataframe(summary)

st.subheader("Filter")
group_view = st.selectbox("Group view", sorted(summary["group_view"].unique()))
model = st.selectbox("Model", sorted(summary["model"].unique()))

filtered = summary[(summary["group_view"] == group_view) & (summary["model"] == model)]
st.dataframe(filtered.sort_values("dp_diff", ascending=False))