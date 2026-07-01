import os
import sys

import streamlit as st
import numpy as np
import pandas as pd

from bakeoff.predict import load_model, predict_from_dict, list_features
from bakeoff.config import load_config
from bakeoff.explain import explain_prediction


MODEL_PATH = "tripod_outputs/final_logreg_firth.pkl"
MODEL_METADATA_KEY = "tripod"


def _feature_input(feature, ftype):
    if ftype == "continuous":
        return st.number_input(feature, value=None, placeholder="Enter value", key=f"cont_{feature}")
    elif ftype == "binary":
        val = st.selectbox(feature, ["", "No (0)", "Yes (1)"], key=f"bin_{feature}")
        return None if val == "" else 1.0 if "Yes" in val else 0.0
    return None


def _get_input_type(feature, metadata):
    if feature in metadata.get("continuous", []):
        return "continuous"
    if feature in metadata.get("binary", []):
        return "binary"
    return "unknown"


def main():
    cfg = load_config()
    tripod_cfg = cfg.get("tripod", {})
    predictors = tripod_cfg.get("pre_specified_predictors", [])
    plausible_bounds = tripod_cfg.get("plausible_bounds", {})
    score_increments = tripod_cfg.get("score_increments", {})

    st.set_page_config(
        page_title="CTO-PCI Risk Predictor (TRIPOD+AI)",
        page_icon="❤️",
        layout="centered",
    )

    st.title("CTO-PCI Risk Predictor")
    st.markdown(
        "Deployable TRIPOD+AI-compliant Firth logistic regression model. "
        "Predicts the likelihood of the composite adverse outcome "
        "(`lv_assist2_aae___2`). Enter one of the 8 pre-specified predictors below."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model not found at `{MODEL_PATH}`. "
            "Run `uv run python -m bakeoff.tripod_main` first to train and save the deployable model."
        )
        st.stop()

    try:
        pipeline, metadata = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    continuous = metadata.get("continuous", [])
    binary = metadata.get("binary", [])

    st.divider()
    st.subheader("Enter 8 Pre-Specified Predictors")

    values = {}

    with st.form("prediction_form"):
        for feature in predictors:
            ftype = _get_input_type(feature, metadata)
            raw = _feature_input(feature, ftype)
            values[feature] = raw

        st.divider()
        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

        st.divider()
        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    if submitted:
        filled = {k: v for k, v in values.items() if v is not None and v != ""}
        total_count = len(predictors)
        filled_count = len(filled)
        with st.spinner("Computing prediction..."):
            try:
                result = predict_from_dict(filled, pipeline, metadata)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            pct = result["probability_positive"] * 100
            st.metric("Risk Score", f"{pct:.1f}%")
        with col2:
            label = "HIGH RISK" if result["prediction"] == 1 else "LOW RISK"
            st.metric("Predicted Class", label)
        with col3:
            st.metric("Fields Used", f"{filled_count} / {total_count}")

        if filled_count < total_count:
            imputed = total_count - filled_count
            st.caption(
                f"{imputed} unfilled field(s) were imputed "
                "(median for continuous, most frequent for binary/categorical)."
            )

        if pct >= 5:
            st.warning(
                "Elevated risk. Consider reviewing the patient's full profile "
                "and discussing with the Heart Team."
            )
        else:
            st.info("Low predicted risk based on entered data.")

        st.divider()
        with st.expander("Explain with AI", expanded=False):
            api_key = st.text_input(
                "OpenAI API key",
                type="password",
                value=os.environ.get("OPENAI_API_KEY", ""),
                key="openai_key",
                help="Stored only for this session. Set OPENAI_API_KEY env var to pre-fill.",
            )
            col_a, _ = st.columns([0.3, 0.7])
            with col_a:
                explain_clicked = st.button("Generate Explanation", key="explain_btn")
            if explain_clicked:
                if not api_key:
                    st.error("Enter an OpenAI API key above.")
                else:
                    with st.spinner("Computing feature contributions and querying LLM..."):
                        try:
                            filled = {k: v for k, v in values.items()
                                      if v is not None and v != ""}
                            expl = explain_prediction(
                                pipeline, filled, metadata, api_key,
                                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                            )
                        except Exception as e:
                            st.error(f"Explanation failed: {e}")
                            st.stop()
                    st.success(expl["explanation"])
                    top = expl.get("top_contributors", [])
                    if top:
                        with st.container(border=True):
                            st.markdown("**Top feature contributors**")
                            for name, val in top:
                                direction = "↑ increases risk" if val > 0 else "↓ decreases risk"
                                st.markdown(f"- `{name}` {direction} (log-odds Δ = {val:.3f})")

    st.divider()
    oof_auc = metadata.get("oof_auc", metadata.get("cv_auc", 0))
    st.caption(
        f"Model: **{metadata.get('model_name', 'LogReg_Firth')}** "
        f"(OOF AUC = {oof_auc:.3f})"
    )


if __name__ == "__main__":
    main()
