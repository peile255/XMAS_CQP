"""
pyexplainer.py

Simplified PyExplainer implementation for RQ1 evaluation.
Compatible with a minimal RuleFit surrogate.
"""

import numpy as np
import pandas as pd


class PyExplainer:
    """
    PyExplainer-style local surrogate explainer.

    Design choice:
    - Uses a simplified RuleFit surrogate
    - Treats each feature as a linear rule
    - Returns top-K contributing features
    """

    def __init__(self, model, X_train, random_state=42):
        """
        Parameters
        ----------
        model : JITModelWrapper
            Frozen black-box prediction model
        X_train : pd.DataFrame
            Training data for surrogate fitting
        random_state : int
        """
        self.model = model
        self.X_train = X_train
        self.random_state = random_state

        self._fit_local_model()

    def _fit_local_model(self):
        """
        Fit the surrogate model on black-box predictions.
        """
        # Black-box predictions
        y_pred = self.model.predict(self.X_train)

        # Lazy import to avoid circular dependency
        from CfExplainer.baselines.rule_fit import RuleFit

        self.explainer = RuleFit(random_state=self.random_state)
        self.explainer.fit(
            self.X_train,
            y_pred,
            feature_names=list(self.X_train.columns)
        )

    def explain(self, x: pd.DataFrame, top_k=5):
        """
        Generate explanation for a single instance.

        Returns
        -------
        list of dict:
            [{"feature": str, "weight": float}, ...]
        """
        x_vec = x.values.flatten()

        # Get (feature, weight) rules
        rules = self.explainer.get_rules(exclude_zero_coef=True)

        # Compute contributions
        contributions = []
        for feature, weight in rules:
            value = float(x[feature].iloc[0])
            contrib = weight * value
            contributions.append((feature, contrib))

        # Rank by absolute contribution
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)

        return [
            {"feature": f, "weight": float(w)}
            for f, w in contributions[:top_k]
        ]
