import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin


class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, max_iter=200, tol=1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def _design(self, X):
        X = np.asarray(X, float)
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X, y):
        y = np.asarray(y, float)
        Xd = self._design(X)
        n, p = Xd.shape
        beta = np.zeros(p)
        for it in range(self.max_iter):
            eta = Xd @ beta
            pr = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
            W = pr * (1 - pr)
            I = (Xd.T * W) @ Xd
            I_inv = np.linalg.pinv(I)
            h = W * np.einsum("ij,jk,ik->i", Xd, I_inv, Xd)
            U = Xd.T @ (y - pr + h * (0.5 - pr))
            step = I_inv @ U
            beta = beta + step
            if np.max(np.abs(step)) < self.tol:
                break
        self.beta_ = beta
        self.coef_ = beta[1:]
        self.intercept_ = beta[0]
        self.vcov_ = I_inv
        self.bse_ = np.sqrt(np.diag(I_inv))
        self.pvals_ = 2 * (1 - stats.norm.cdf(np.abs(beta / self.bse_)))
        self.ci_ = np.column_stack(
            [beta - 1.96 * self.bse_, beta + 1.96 * self.bse_]
        )
        self.classes_ = np.array([0, 1])
        self.n_iter_ = it + 1
        return self

    def decision_function(self, X):
        return self._design(X) @ self.beta_

    def predict_proba(self, X):
        pr = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - pr, pr])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
