import numpy as np
import pandas as pd

from bakeoff.firth import FirthLogisticRegression


def _get_feature_names(metadata):
    return metadata.get("feature_names", metadata.get("predictors", []))


def _get_contributions_firth(pipeline, input_row_df, metadata):
    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["prep"]
    Xt = prep.transform(input_row_df)
    feat = list(prep.get_feature_names_out())
    contribs = {}
    for i, name in enumerate(feat):
        val = float(Xt[0, i])
        beta = float(model.coef_[i])
        contribs[name] = beta * val
    return contribs


def _get_contributions_nb(pipeline, input_row_df, metadata):
    from sklearn.preprocessing import OneHotEncoder
    nb = pipeline.named_steps["model"]
    prep = pipeline.named_steps["prep"]
    Xt = prep.transform(input_row_df)
    feat = list(prep.get_feature_names_out())
    binarized = (Xt > 0.5).astype(int).ravel()
    log_prob = nb.feature_log_prob_
    contribs = {}
    for i, name in enumerate(feat):
        if binarized[i] == 1:
            contrib = float(log_prob[1, i] - log_prob[0, i])
        else:
            p1 = float(np.exp(log_prob[1, i]))
            p0 = float(np.exp(log_prob[0, i]))
            contrib = float(np.log(1 - p1 + 1e-10) - np.log(1 - p0 + 1e-10))
        contribs[name] = contrib
    return contribs


def _compute_contributions(pipeline, input_row_df, metadata):
    model = pipeline.named_steps["model"]
    if isinstance(model, FirthLogisticRegression):
        return _get_contributions_firth(pipeline, input_row_df, metadata)
    else:
        return _get_contributions_nb(pipeline, input_row_df, metadata)


def _extract_impute_values(pipeline, metadata):
    prep = pipeline.named_steps["prep"]
    values = {}
    for tf_key in ["cont", "bin", "cat"]:
        tf = prep.named_transformers_.get(tf_key)
        if tf is None:
            continue
        if tf_key == "cont":
            imp = tf.named_steps["imp"]
            cols = metadata.get("continuous", [])
            strategy = "median"
        elif tf_key == "bin":
            imp = tf
            cols = metadata.get("binary", [])
            strategy = "most_frequent"
        else:
            imp = tf.named_steps["imp"]
            cols = metadata.get("categorical", [])
            strategy = "most_frequent"
        if not cols or not hasattr(imp, "statistics_") or imp.statistics_ is None:
            continue
        for i, col in enumerate(cols):
            if i < len(imp.statistics_):
                values[col] = (
                    float(imp.statistics_[i]) if strategy == "median"
                    else str(imp.statistics_[i])
                )
    return values


def _build_explanation(contributions, input_values, probability, top_n=8):
    sorted_items = sorted(contributions.items(), key=lambda x: -abs(x[1]))
    positive = []
    negative = []
    for name, val in sorted_items:
        if val > 0:
            actual = input_values.get(name, "N/A")
            increase = np.exp(val)
            positive.append(f"{name}={actual} (risk increased ~{increase:.1f}x)")
        else:
            actual = input_values.get(name, "N/A")
            decrease = np.exp(-val)
            negative.append(f"{name}={actual} (risk decreased ~{decrease:.1f}x)")
        if len(positive) + len(negative) >= top_n * 2:
            break
    prompt = (
        f"The model predicts a {probability*100:.1f}% risk of the adverse outcome."
    )
    if positive:
        prompt += f" Top factors increasing risk: {'; '.join(positive[:top_n])}."
    if negative:
        prompt += f" Top factors decreasing risk: {'; '.join(negative[:top_n])}."
    prompt += (
        " In 2-3 plain sentences, summarize the main drivers of this prediction "
        "and what clinical picture they suggest."
    )
    return prompt


def _call_openai(prompt, api_key, model="gpt-4o-mini", max_tokens=300):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a cardiology decision-support assistant. "
                "Explain model predictions in 2-3 plain sentences for clinicians. "
                "Be specific about which features drive risk up or down.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def explain_prediction(pipeline, input_dict, metadata, api_key, model="gpt-4o-mini"):
    feature_names = _get_feature_names(metadata)
    row = {c: input_dict.get(c, np.nan) for c in feature_names}
    input_df = pd.DataFrame([row])
    probability = float(pipeline.predict_proba(input_df)[0, 1])
    contributions = _compute_contributions(pipeline, input_df, metadata)
    impute_values = _extract_impute_values(pipeline, metadata)
    all_inputs = {}
    for c in feature_names:
        raw = input_dict.get(c, impute_values.get(c, np.nan))
        all_inputs[c] = raw
    prompt = _build_explanation(contributions, all_inputs, probability)
    explanation = _call_openai(prompt, api_key, model=model)
    top_contributors = (
        sorted(contributions.items(), key=lambda x: -abs(x[1]))[:10]
    )
    return {
        "explanation": explanation,
        "probability": probability,
        "top_contributors": top_contributors,
    }
