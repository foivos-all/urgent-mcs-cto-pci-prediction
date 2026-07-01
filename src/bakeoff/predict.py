import numpy as np
import pandas as pd
import joblib


def load_model(model_path="tripod_outputs/final_logreg_firth.pkl"):
    data = joblib.load(model_path)
    return data["pipeline"], data["metadata"]


def predict_from_dict(values, pipeline, metadata, full_default=np.nan):
    feature_names = metadata.get("feature_names", metadata.get("predictors", []))
    if not feature_names:
        binary = metadata.get("binary", [])
        continuous = metadata.get("continuous", [])
        feature_names = binary + continuous
    row = {c: full_default for c in feature_names}
    row.update(values)
    df = pd.DataFrame([row])
    proba = pipeline.predict_proba(df)[0]
    pred_class = int(proba[1] >= 0.5)
    return {
        "prediction": pred_class,
        "probability_negative": float(proba[0]),
        "probability_positive": float(proba[1]),
    }


def list_features(metadata):
    all_f = metadata.get("feature_names", metadata.get("predictors", []))
    return {
        "all": all_f,
        "binary": metadata.get("binary", []),
        "categorical": metadata.get("categorical", []),
        "continuous": metadata.get("continuous", []),
    }
