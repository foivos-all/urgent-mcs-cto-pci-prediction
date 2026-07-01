import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _intset(s):
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().mean() < 0.5:
        return None
    vv = sn.dropna().unique()
    if not np.all(np.isclose(np.mod(vv, 1.0), 0.0, atol=1e-6)):
        return None
    return set(np.round(vv).astype(int))


def recode_binary(s):
    iset = _intset(s)
    sn = pd.to_numeric(s, errors="coerce")
    if iset == {1, 2}:
        return sn.map({1: 1.0, 2: 0.0}), "1=yes\u21921, 2=no\u21920"
    if iset == {1, 2, 3}:
        return sn.map({1: 1.0, 2: 0.0, 3: np.nan}), "1=yes\u21921, 2=no\u21920, 3=na\u2192missing"
    if iset == {0, 1}:
        return sn.astype(float), ""
    codes, uniques = pd.factorize(s, sort=True)
    note = f"{uniques[0]}\u21920, {uniques[1]}\u21921" if len(uniques) >= 2 else ""
    return pd.Series(np.where(codes == -1, np.nan, codes).astype(float), index=s.index), note


def classify(s, cat_max, recode123):
    nun = s.dropna().nunique()
    if nun <= 1:
        return "DROPPED", "constant"
    iset = _intset(s)
    numeric = pd.to_numeric(s, errors="coerce").notna().mean() >= 0.5
    if nun == 2:
        return "binary", ""
    if recode123 and iset is not None and iset <= {1, 2, 3} and {1, 2} <= iset:
        return "binary", ""
    if numeric and iset is not None and len(iset) < cat_max:
        return "categorical", ""
    if numeric:
        return "continuous", ""
    if nun < cat_max:
        return "categorical", ""
    return "DROPPED", f"high-cardinality text ({nun} levels)"


def load_data(data_path, target):
    df = pd.read_csv(data_path, encoding="latin1")
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        print("Dropping index-like columns:", unnamed)
    df = df.drop(columns=unnamed)
    df = df.loc[~df[target].isna()].copy()
    y = df[target]
    if not np.issubdtype(y.dtype, np.number):
        y = pd.Categorical(y).codes
    y = pd.Series(y, index=df.index).astype(int).reset_index(drop=True)
    X0 = df.drop(columns=[target]).reset_index(drop=True)
    all_missing = X0.columns[X0.isna().all()].tolist()
    dup = X0.columns[X0.astype(str).T.duplicated().values].tolist()
    drop_junk = sorted(set(all_missing) | set(dup))
    if drop_junk:
        print(f"Dropping {len(drop_junk)} all-missing/duplicate columns")
    X0 = X0.drop(columns=drop_junk)
    print(f"Rows: {len(y)} | events: {int(y.sum())} ({y.mean():.3%}) | columns left to classify: {X0.shape[1]}")
    return X0, y


def classify_variables(
    X0,
    cat_max=20,
    recode123=True,
    yesno_na_vars=None,
    force_types=None,
):
    if yesno_na_vars is None:
        yesno_na_vars = []
    if force_types is None:
        force_types = {}
    binary, categorical, continuous, dropped = [], [], [], []
    X = pd.DataFrame(index=X0.index)
    note_of = {}
    for c in X0.columns:
        if c in yesno_na_vars:
            X[c] = pd.to_numeric(X0[c], errors="coerce").map({1: 1.0, 2: 0.0})
            binary.append(c)
            note_of[c] = "forced 1=yes\u21921, 2=no\u21920, else\u2192missing"
            continue
        t, why = classify(X0[c], cat_max, recode123)
        if c in force_types:
            t, why = force_types[c], "forced"
        if t == "binary":
            rec, note = recode_binary(X0[c])
            X[c] = rec
            binary.append(c)
            note_of[c] = note or why
        elif t == "categorical":
            X[c] = X0[c]
            categorical.append(c)
            note_of[c] = why
        elif t == "continuous":
            X[c] = pd.to_numeric(X0[c], errors="coerce")
            continuous.append(c)
            note_of[c] = why
        else:
            dropped.append((c, why))
            note_of[c] = why
    type_of = (
        {c: "binary" for c in binary}
        | {c: "categorical" for c in categorical}
        | {c: "continuous" for c in continuous}
    )
    rows = []
    for c in X0.columns:
        s = X0[c]
        lv = int(s.dropna().nunique())
        samp = ", ".join(
            map(str, sorted(pd.Series(s.dropna().unique()).tolist(), key=lambda v: str(v))[:6])
        )
        rows.append({
            "variable": c,
            "type": type_of.get(c, "DROPPED"),
            "levels": lv,
            "sample_values": samp,
            "note": note_of.get(c, ""),
        })
    rank = {"binary": 0, "categorical": 1, "continuous": 2, "DROPPED": 3}
    typing_table = (
        pd.DataFrame(rows)
        .sort_values(
            ["type", "variable"],
            key=lambda s: s.map(rank) if s.name == "type" else s,
        )
        .reset_index(drop=True)
    )
    print(f"binary={len(binary)}  categorical={len(categorical)}  continuous={len(continuous)}  dropped={len(dropped)}\n")
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 46, "display.width", 170):
        print(typing_table.to_string(index=False))
    return X, binary, categorical, continuous, dropped, typing_table


def reduce_redundancy(
    X_train,
    X_test,
    y_train,
    redundant_groups,
    binary,
    categorical,
    continuous,
    random_state=42,
):
    drop_redundant = []
    print("Redundancy reduction (univariate train AUC; keep best per group):")
    for grp in redundant_groups:
        present = [c for c in grp if c in X_train.columns]
        if len(present) <= 1:
            if present:
                print(f"  {grp}: only {present} present \u2014 kept")
            continue
        scored = sorted(
            ((uni_auc(X_train[c], y_train), c) for c in present), reverse=True
        )
        keep = scored[0][1]
        drop_redundant += [c for _, c in scored[1:]]
        print(f"  group: {present}")
        for a, c in scored:
            tag = "<- KEEP" if c == keep else "drop"
            print(f"      {c:34s} AUC={a:.3f}  {tag}")
    if drop_redundant:
        X_train = X_train.drop(columns=drop_redundant)
        X_test = X_test.drop(columns=drop_redundant)
        binary = [c for c in binary if c not in drop_redundant]
        categorical = [c for c in categorical if c not in drop_redundant]
        continuous = [c for c in continuous if c not in drop_redundant]
        print(f"\nDropped {len(drop_redundant)} redundant column(s): {drop_redundant}")
    else:
        print("  (no groups had >1 member present in the data)")
    print(f"Predictors entering the models: {len(binary)+len(categorical)+len(continuous)}")
    return X_train, X_test, binary, categorical, continuous


def uni_auc(x, yv):
    x = pd.to_numeric(x, errors="coerce")
    x = x.fillna(x.median())
    if x.nunique() < 2:
        return 0.5
    a = roc_auc_score(yv, x)
    return max(a, 1 - a)


def train_val_split(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
