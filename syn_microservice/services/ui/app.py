import os, time
import requests
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")

st.set_page_config(page_title="Synthetic Health Dashboard", layout="wide")
st.title("Synthetic Health Data Dashboard (SDV + Fairness + Privacy)")

def safe_json(resp: requests.Response):
    """Return (data, err_msg). data is dict/list or None."""
    try:
        return resp.json(), None
    except Exception as e:
        text = resp.text or ""
        snippet = text[:500].replace("\n", "\\n")
        return None, f"JSON parse failed: {e}. status={resp.status_code}, content-type={resp.headers.get('content-type')}, body[:500]={snippet}"

def get_json(url: str, timeout: int = 10):
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        return None, None, f"Request failed: {e}"

    if resp.status_code != 200:
        # not OK — don’t attempt resp.json() blindly
        text = resp.text or ""
        snippet = text[:500].replace("\n", "\\n")
        return None, resp, f"HTTP {resp.status_code}. content-type={resp.headers.get('content-type')}, body[:500]={snippet}"

    data, err = safe_json(resp)
    return data, resp, err


with st.sidebar:
    st.header("Run Config")

    dataset = st.text_input("Dataset name", "diabetes")

    generators = st.multiselect(
        "Generators",
        ["gaussian_copula", "ctgan", "tvae"],
        default=["gaussian_copula", "ctgan", "tvae"]
    )

    num_rows = st.number_input("Synthetic rows per model", min_value=100, max_value=200000, value=5000, step=100)

    st.divider()
    st.subheader("Speed knobs")
    subset_rows = st.number_input(
        "Subset rows (real_train)",
        min_value=0, max_value=100000, value=10000, step=1000,
        help="0 = use full training split (slow)."
    )
    pair_metric_rows = st.number_input("Pair metric subsample rows", min_value=500, max_value=20000, value=5000, step=500)

    st.divider()
    st.subheader("Privacy knobs")
    privacy_max_n = st.number_input("Privacy sample size", min_value=500, max_value=20000, value=5000, step=500)
    privacy_percentile = st.slider("NN leakage percentile", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    st.divider()
    save_plots = st.checkbox("Save HTML plots into artifacts", value=True)

    st.caption(f"API_URL: {API_URL}")

    if st.button("Run Experiment"):
        payload = {
            "dataset": dataset,
            "generators": generators,
            "num_rows": int(num_rows),
            "subset_rows": int(subset_rows),
            "pair_metric_rows": int(pair_metric_rows),
            "privacy_max_n": int(privacy_max_n),
            "privacy_percentile": float(privacy_percentile),
            "save_plots": bool(save_plots),
        }

        try:
            resp = requests.post(f"{API_URL}/runs", json=payload, timeout=30)
            if resp.status_code != 200:
                st.error(f"POST /runs failed: {resp.status_code} {resp.text[:500]}")
            else:
                data, err = safe_json(resp)
                if err:
                    st.error(err)
                else:
                    st.session_state["run_id"] = data["run_id"]
        except Exception as e:
            st.error(f"POST /runs request failed: {e}")


run_id = st.session_state.get("run_id")
if not run_id:
    st.info("Click **Run Experiment** to start a job.")
    st.stop()

st.subheader(f"Run: {run_id}")

status_box = st.empty()
results_box = st.empty()

for _ in range(180):  
    info, resp, err = get_json(f"{API_URL}/runs/{run_id}", timeout=10)
    if err:
        status_box.error(f"Could not read run status. {err}")
        break

    status = info.get("status", "unknown")
    status_box.write({"status": status, **{k: info[k] for k in info if k != "job_json"}})

    if status == "failed":
        st.error("Run failed. Check artifacts/<run_id>/error.txt")
        break

    if status == "done":
        res_resp = None
        try:
            res_resp = requests.get(f"{API_URL}/runs/{run_id}/results", timeout=30)
        except Exception as e:
            st.error(f"GET /results request failed: {e}")
            break

        if res_resp.status_code == 404:
            # results not ready yet; keep polling briefly
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
            st.warning("No rows in results payload.")

        st.markdown("### Baseline (Real → Real)")
        st.json(full.get("baseline", {}))

        st.markdown("### Per-model Details")
        model_names = list(full.get("models", {}).keys())
        if model_names:
            chosen = st.selectbox("Select model", model_names, index=0)
            st.json(full["models"][chosen])
        break

    time.sleep(1)
else:
    st.warning("Still running... check again in a moment.")
