import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


def plot_variable_classification(binary, categorical, continuous, dropped, output_dir):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(
        ["binary", "categorical", "continuous", "dropped"],
        [len(binary), len(categorical), len(continuous), len(dropped)],
        color=["#4da3ff", "#e6c84d", "#39c08c", "#8aa0b6"],
    )
    ax.set_ylabel("columns")
    ax.set_title("Variable classification")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/variable_classification.png", dpi=150)
    plt.close(fig)


def plot_fixed_k_bars(fixed_df, order, fixed_ks, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, K in zip(axes, fixed_ks):
        d = (
            fixed_df[fixed_df.K == K]
            .set_index("model")
            .reindex(order)
            .dropna(subset=["cv_auc"])
        )
        xp = np.arange(len(d))
        w = 0.38
        ax.bar(xp - w / 2, d["cv_auc"], w, label="train (CV)", color="#4da3ff")
        te_err = np.vstack(
            [d["test_auc"] - d["test_lo"], d["test_hi"] - d["test_auc"]]
        )
        ax.bar(
            xp + w / 2, d["test_auc"], w, yerr=te_err, capsize=2,
            label="test (95% CI)", color="#ff9e5a",
        )
        ax.axhline(0.5, ls=":", c="grey", lw=1)
        ax.axhline(0.80, ls="--", c="crimson", lw=1)
        ax.set_xticks(xp)
        ax.set_xticklabels(d.index, rotation=45, ha="right")
        ax.set_title(f"K = {K}")
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(loc="lower right")
    fig.suptitle("Fixed-K models \u2014 train (cross-validated) vs held-out test AUC")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fixed_k_bars.png", dpi=150)
    plt.close(fig)


def plot_fixed_k_roc(fixed_test, order, fixed_ks, y_test, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)
    for ax, K in zip(axes, fixed_ks):
        for name in order:
            if (name, K) in fixed_test:
                s = fixed_test[(name, K)]
                fpr, tpr, _ = roc_curve(y_test, s)
                ax.plot(
                    fpr, tpr, lw=1.5,
                    label=f"{name} ({roc_auc_score(y_test, s):.3f})",
                )
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        ax.set_title(f"test ROC \u2014 K = {K}")
        ax.set_xlabel("1 - specificity")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("sensitivity")
    fig.suptitle("Held-out test ROC by model, at each fixed K")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fixed_k_roc.png", dpi=150)
    plt.close(fig)


def plot_best_k_curves(k_curves, k_grid, order, output_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    xs = [str(k) for k in k_grid]
    for name in order:
        kc = k_curves[name].reindex(xs)
        ax.plot(xs, kc.values, "o-", lw=1.8, label=name)
    ax.set_xlabel("number of features (K) selected in-fold")
    ax.set_ylabel("cross-validated ROC-AUC (best HP at each K)")
    ax.set_title("Best K per model")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/best_k_curves.png", dpi=150)
    plt.close(fig)


def plot_train_test_auc(res, output_dir):
    order = res["model"].tolist()
    xpos = np.arange(len(order))
    w = 0.38
    cv_a = res.set_index("model").loc[order, "cv_auc"].values
    cv_e = res.set_index("model").loc[order, "cv_sd"].values
    te_a = res.set_index("model").loc[order, "test_auc"].values
    te_lo = res.set_index("model").loc[order, "test_lo"].values
    te_hi = res.set_index("model").loc[order, "test_hi"].values
    te_err = np.vstack([te_a - te_lo, te_hi - te_a])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        xpos - w / 2, cv_a, w, yerr=cv_e, capsize=3,
        label="training (cross-validated)", color="#4da3ff",
    )
    ax.bar(
        xpos + w / 2, te_a, w, yerr=te_err, capsize=3,
        label="held-out test (95% CI)", color="#ff9e5a",
    )
    ax.axhline(0.5, ls=":", color="grey", lw=1)
    ax.axhline(0.80, ls="--", color="crimson", lw=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("AUC for best settings of each model \u2014 training vs testing")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/train_test_auc.png", dpi=150)
    plt.close(fig)


def plot_roc_best(res, oof_scores, test_scores, y_train, y_test, output_dir):
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for name in res["model"]:
        fpr, tpr, _ = roc_curve(y_train, oof_scores[name])
        ax[0].plot(
            fpr, tpr, lw=1.6,
            label=f"{name} ({roc_auc_score(y_train, oof_scores[name]):.3f})",
        )
        fpr, tpr, _ = roc_curve(y_test, test_scores[name])
        ax[1].plot(
            fpr, tpr, lw=1.6,
            label=f"{name} ({roc_auc_score(y_test, test_scores[name]):.3f})",
        )
    for a, title in zip(ax, ["Training (out-of-fold)", "Held-out test"]):
        a.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        a.set_xlabel("1 - specificity")
        a.set_ylabel("sensitivity")
        a.set_title(f"ROC \u2014 {title}")
        a.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/roc_best_settings.png", dpi=150)
    plt.close(fig)


def plot_pr_curve(res, test_scores, y_test, output_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name in res["model"].head(6):
        pr, rc, _ = precision_recall_curve(y_test, test_scores[name])
        ax.plot(
            rc, pr, lw=1.8,
            label=f"{name} (AP {average_precision_score(y_test, test_scores[name]):.3f})",
        )
    ax.axhline(
        y_test.mean(), color="grey", ls=":", lw=1,
        label=f"baseline {y_test.mean():.3f}",
    )
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision\u2013Recall (test) \u2014 top 6 by CV")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/precision_recall_curve.png", dpi=150)
    plt.close(fig)


def generate_all_plots(
    output_dir,
    binary,
    categorical,
    continuous,
    dropped,
    fixed_df,
    fixed_test,
    order,
    k_curves,
    k_grid,
    res,
    oof_scores,
    test_scores,
    y_train,
    y_test,
):
    plot_variable_classification(binary, categorical, continuous, dropped, output_dir)
    plot_fixed_k_bars(fixed_df, order, [15, 25, 50], output_dir)
    plot_fixed_k_roc(fixed_test, order, [15, 25, 50], y_test, output_dir)
    plot_best_k_curves(k_curves, k_grid, order, output_dir)
    plot_train_test_auc(res, output_dir)
    plot_roc_best(res, oof_scores, test_scores, y_train, y_test, output_dir)
    plot_pr_curve(res, test_scores, y_test, output_dir)
