import os

import streamlit as st

from bakeoff.config import load_config
from bakeoff.explain import explain_prediction
from bakeoff.predict import load_model, predict_from_dict


MODEL_PATH = "tripod_outputs/final_logreg_firth.pkl"

DISPLAY_NAMES = {
    "retro": "Retrograde-first Strategy",
    "calcification_med_sev": "Moderate/Severe Calcification",
    "peripheral_arterial_diseas": "Peripheral Arterial Disease",
    "acs": "Acute Coronary Syndrome Presentation",
    "proximal_cap_ambiguity": "Proximal-cap Ambiguity",
    "left_ventr_ejection_fract": "Left Ventricular Ejection Fraction (%)",
    "age_manual_input": "Age (years)",
    "occlusion_length_mm": "Occlusion Length (mm)",
}


def _feature_input(feature, label, ftype):
    if ftype == "continuous":
        return st.number_input(label, value=None, placeholder="Enter value", key=f"cont_{feature}")
    elif ftype == "binary":
        val = st.selectbox(label, ["No (0)", "Yes (1)"], key=f"bin_{feature}")
        return 0.0 if "No" in val else 1.0
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

    st.set_page_config(
        page_title="PROGRESS-uMCS Score",
        page_icon="❤️",
        layout="centered",
    )

    st.title("PROGRESS-uMCS Score")
    st.markdown(
        "Predicts the likelihood of requiring cardiac support during CTO PCI."
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

    st.divider()
    st.subheader("Enter 8 Pre-Specified Predictors")

    values = {}

    with st.form("prediction_form"):
        for feature in predictors:
            ftype = _get_input_type(feature, metadata)
            label = DISPLAY_NAMES.get(feature, feature)
            raw = _feature_input(feature, label, ftype)
            values[feature] = raw

        st.divider()
        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    if submitted:
        missing = [k for k, v in values.items() if v is None or v == ""]
        if missing:
            st.error("Please fill in all fields before predicting.")
            st.stop()
        filled = values
        with st.spinner("Computing prediction..."):
            try:
                result = predict_from_dict(filled, pipeline, metadata)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.session_state.result = result
        st.session_state.filled = filled
        st.session_state.form_values = values

    if "result" in st.session_state:
        result = st.session_state.result
        filled = st.session_state.filled
        values = st.session_state.form_values

        pct = result["probability_positive"] * 100

        st.divider()
        st.metric("Risk Score", f"{pct:.1f}%")

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
