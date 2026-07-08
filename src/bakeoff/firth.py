import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """Firth penalized logistic regression (Firth 1993; Heinze & Schemper 2002) with optional
    FLIC / FLAC predicted-probability calibration (Puhr et al., Stat Med 2017) and optional
    uniform-shrinkage recalibration. Odds ratios are always the Firth (de-biased) estimates,
    stored in beta_firth_/ci_/pvals_ regardless of variant."""

    def __init__(self, max_iter=200, tol=1e-8, variant="firth", shrinkage=1.0):
        self.max_iter = max_iter
        self.tol = tol
        self.variant = variant
        self.shrinkage = shrinkage

    def _design(self, X):
        X = np.asarray(X, float)
        return np.column_stack([np.ones(len(X)), X])

    def _firth_fit(self, Xd, y):
        n, p = Xd.shape
        beta = np.zeros(p)
        h = np.zeros(n)
        it = 0
        for it in range(self.max_iter):
            eta = Xd @ beta
            pr = np.clip(_sigmoid(eta), 1e-12, 1 - 1e-12)
            W = pr * (1 - pr)
            I = (Xd.T * W) @ Xd
            I_inv = np.linalg.pinv(I)
            h = W * np.einsum("ij,jk,ik->i", Xd, I_inv, Xd)
            U = Xd.T @ (y - pr + h * (0.5 - pr))
            step = I_inv @ U
            beta = beta + step
            if np.max(np.abs(step)) < self.tol:
                break
        return beta, I_inv, h, it + 1

    def _ml_intercept(self, offset, y, b0=0.0):
        for _ in range(200):
            p = _sigmoid(b0 + offset)
            g = np.sum(y - p)
            H = -np.sum(p * (1 - p))
            if abs(H) < 1e-12:
                break
            new = b0 - g / H
            if abs(new - b0) < 1e-10:
                b0 = new
                break
            b0 = new
        return b0

    def fit(self, X, y):
        y = np.asarray(y, float)
        Xd = self._design(X)
        n, p = Xd.shape
        beta, I_inv, h, nit = self._firth_fit(Xd, y)

        # Firth inference (kept for the OR table regardless of variant)
        self.beta_firth_ = beta.copy()
        self.vcov_ = I_inv
        self.bse_ = np.sqrt(np.diag(I_inv))
        self.pvals_ = 2 * (1 - stats.norm.cdf(np.abs(beta / self.bse_)))
        self.ci_ = np.column_stack([beta - 1.96 * self.bse_, beta + 1.96 * self.bse_])
        self.hat_ = h
        self.n_iter_ = nit

        if self.variant == "firth":
            final = beta.copy()
        elif self.variant == "flic":
            offset = Xd[:, 1:] @ beta[1:]
            b0 = self._ml_intercept(offset, y, beta[0])
            final = beta.copy()
            final[0] = b0
        elif self.variant == "flac":
            Xs = Xd[:, 1:]
            Xa = np.vstack([Xs, Xs, Xs])
            ga = np.concatenate([np.zeros(n), np.ones(n), np.ones(n)])
            ya = np.concatenate([y, np.ones(n), np.zeros(n)])
            wa = np.concatenate([np.ones(n), h / 2, h / 2])
            Da = sm.add_constant(np.column_stack([Xa, ga]))
            m = sm.GLM(ya, Da, family=sm.families.Binomial(), freq_weights=wa).fit()
            par = np.asarray(m.params)
            final = np.concatenate([[par[0]], par[1:1 + Xs.shape[1]]])  # drop g
        else:
            raise ValueError("variant must be firth|flic|flac")

        if self.shrinkage != 1.0:  # deployment recalibration
            sl = final[1:] * self.shrinkage
            offset = Xd[:, 1:] @ sl
            b0 = self._ml_intercept(offset, y, final[0])
            final = np.concatenate([[b0], sl])

        self.beta_ = final
        self.coef_ = final[1:]
        self.intercept_ = final[0]
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self._design(X) @ self.beta_

    def predict_proba(self, X):
        pr = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - pr, pr])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
