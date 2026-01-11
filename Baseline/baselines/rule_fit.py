"""
rule_fit.py

A minimal RuleFit-style surrogate model for PyExplainer.
Designed for:
- determinism
- interface compatibility
- RQ1 stability evaluation
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class RuleFit:
    """
    Simplified RuleFit surrogate:
    - Each feature is treated as a linear rule
    - Coefficients indicate rule strength
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            random_state=random_state,
            max_iter=1000,
        )
        self.feature_names = None
        self.coef_ = None

    def fit(self, X, y, feature_names=None):
        """
        Fit surrogate model.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
        y : array-like
        feature_names : list[str], optional
        """
        if hasattr(X, "values"):
            X = X.values

        self.model.fit(X, y)
        self.coef_ = self.model.coef_[0]

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]

        return self

    def get_rules(self, exclude_zero_coef=True):
        """
        Return rules as (feature_name, weight).
        This mimics the interface expected by PyExplainer.
        """
        rules = []

        for name, weight in zip(self.feature_names, self.coef_):
            if exclude_zero_coef and weight == 0:
                continue
            rules.append((name, float(weight)))

        return rules

    def explain_instance(self, x, top_k=5):
        """
        Rank rules by contribution for a given instance.
        """
        contributions = self.coef_ * x
        idx = np.argsort(np.abs(contributions))[::-1][:top_k]

        return [
            (self.feature_names[i], float(contributions[i]))
            for i in idx
            if contributions[i] != 0
        ]
