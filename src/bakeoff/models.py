import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from xgboost import XGBClassifier


def model_zoo(y_train, fast_mode=False):
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    spw = float(cw[0])  # weight for positive class
    z = {
        "LogReg": (
            LogisticRegression(
                solver="liblinear", class_weight="balanced", max_iter=5000, random_state=42
            ),
            {"model__penalty": ["l1", "l2"], "model__C": [0.01, 0.1, 1.0]},
        ),
        "NB_Gaussian": (
            GaussianNB(),
            {"model__var_smoothing": [1e-9, 1e-7, 1e-5]},
        ),
        "NB_Bernoulli": (
            BernoulliNB(),
            {
                "model__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                "model__fit_prior": [True, False],
                "model__binarize": [0.0, 0.5],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [5, 15, 31], "model__weights": ["uniform", "distance"]},
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss",
                n_estimators=400,
                tree_method="hist",
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1,
            ),
            {
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.1],
                "model__reg_lambda": [1.0, 5.0],
                "model__reg_alpha": [0.0, 1.0],
                "model__scale_pos_weight": [float(np.sqrt(spw)), spw],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=400, class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ),
            {"model__max_depth": [6, 10], "model__min_samples_leaf": [5, 20]},
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(
                n_estimators=400, class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ),
            {"model__max_depth": [6, 10], "model__min_samples_leaf": [5, 20]},
        ),
        "AdaBoost": (
            AdaBoostClassifier(n_estimators=200, random_state=42),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.5, 1.0]},
        ),
        "HistGBM": (
            HistGradientBoostingClassifier(
                max_iter=400, class_weight="balanced", random_state=42
            ),
            {
                "model__learning_rate": [0.03, 0.1],
                "model__l2_regularization": [0.0, 1.0, 5.0],
            },
        ),
    }
    if not fast_mode:
        z["SVM"] = (
            SVC(probability=True, class_weight="balanced", random_state=42),
            {"model__C": [1.0, 10.0], "model__gamma": ["scale"]},
        )
        z["MLP"] = (
            MLPClassifier(max_iter=400, early_stopping=True, n_iter_no_change=10, random_state=42),
            {
                "model__alpha": [1e-4, 1e-2],
                "model__hidden_layer_sizes": [(32, 16), (64, 32, 16)],
            },
        )
    return z, spw
