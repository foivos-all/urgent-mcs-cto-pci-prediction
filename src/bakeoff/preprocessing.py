from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


def build_preprocessor(binary, categorical, continuous):
    return ColumnTransformer(
        [
            (
                "cont",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                ]),
                continuous,
            ),
            ("bin", SimpleImputer(strategy="most_frequent"), binary),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_cv(cv_splits, random_state):
    return StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)


def get_k_grid(prep, X_train, k_grid_config):
    n_enc = prep.fit(X_train).transform(X_train.head(50)).shape[1]
    k_grid = [k for k in k_grid_config if k == "all" or k < n_enc]
    if "all" not in k_grid:
        k_grid.append("all")
    print(f"Encoded feature count: {n_enc} | K grid: {k_grid}")
    return k_grid, n_enc
