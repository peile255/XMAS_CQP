"""
cfexplainer.py

Deterministic, rule-based CfExplainer baseline
for RQ1 run-to-run stability evaluation.

Design goals:
- model-centric
- deterministic given random_state
- explanation depends ONLY on model-visible features
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class CfExplainer:
    """
    Simplified CfExplainer:
    - Fit a shallow surrogate decision tree
    - Extract feature importance as explanation
    """

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        random_state: int = 42,
        max_depth: int = 3,
    ):
        self.model = model
        self.X_train = X_train
        self.random_state = random_state
        self.feature_names = list(X_train.columns)

        # --- get model predictions as pseudo-labels ---
        y_pred = model.predict(X_train)

        # --- train surrogate tree ---
        self.surrogate = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
        )
        self.surrogate.fit(X_train, y_pred)

        self.importances = self.surrogate.feature_importances_

    # --------------------------------------------------
    # Main API
    # --------------------------------------------------

    def explain(self, x_df: pd.DataFrame, top_k: int = 5):
        """
        Generate explanation for a single instance.

        Returns:
        [
          {
            "feature": str,
            "importance": float
          },
          ...
        ]
        """

        x = x_df[self.feature_names].values[0]

        contributions = self.importances * x
        idx = np.argsort(np.abs(contributions))[::-1]

        rules = []
        for i in idx[:top_k]:
            if self.importances[i] == 0:
                continue

            rules.append({
                "feature": self.feature_names[i],
                "importance": float(contributions[i]),
            })

        return rules
