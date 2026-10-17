"""Feature engineering module for credit risk modeling.

Provides domain-specific financial ratios and transformations wrapped in a
scikit-learn compatible transformer to prevent data leakage and guarantee
seamless pipeline integration.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-specific feature engineering transformer for credit risk datasets.

    Extracts financial leverage, burden, and duration ratios only when
    underlying source columns exist in the input dataframe.

    Parameters
    ----------
    include_ratios : bool, default=True
        Whether to compute financial interaction ratios.
    """

    def __init__(self, include_ratios: bool = True):
        self.include_ratios = include_ratios
        self.created_features_: List[str] = []
        self.feature_names_in_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditFeatureEngineer":
        """Store input feature names and validate structure.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : Optional[pd.Series], default=None
            Target values (ignored).

        Returns
        -------
        CreditFeatureEngineer
            Fitted transformer instance.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(getattr(X, "shape", [0, 0])[1])]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create domain financial features without mutating the original dataframe.

        Parameters
        ----------
        X : pd.DataFrame
            Features dataframe.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe with augmented feature columns.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_ if self.feature_names_in_ else None)

        X_out = X.copy()
        created: List[str] = []

        if not self.include_ratios:
            self.created_features_ = created
            return X_out

        cols_lower = {str(c).lower(): c for c in X_out.columns}

        # 1. Monthly installment burden (Credit Amount / Duration)
        credit_col = cols_lower.get("credit_amount")
        duration_col = cols_lower.get("duration")
        if credit_col and duration_col:
            credit_vals = pd.to_numeric(X_out[credit_col], errors="coerce").fillna(0.0)
            duration_vals = pd.to_numeric(X_out[duration_col], errors="coerce").fillna(1.0)
            duration_safe = np.maximum(duration_vals.values, 1.0)
            
            # Monthly estimated principal burden
            X_out["fe_monthly_credit_burden"] = credit_vals.values / duration_safe
            created.append("fe_monthly_credit_burden")

            # Credit duration risk product: high credit over prolonged duration poses compounded default exposure
            X_out["fe_credit_duration_exposure"] = (credit_vals.values / 1000.0) * (duration_safe / 12.0)
            created.append("fe_credit_duration_exposure")

        # 2. Credit to Age ratio (leverage relative to life stage)
        age_col = cols_lower.get("age")
        if credit_col and age_col:
            credit_vals = pd.to_numeric(X_out[credit_col], errors="coerce").fillna(0.0)
            age_vals = pd.to_numeric(X_out[age_col], errors="coerce").fillna(18.0)
            age_safe = np.maximum(age_vals.values, 18.0)
            
            X_out["fe_credit_to_age_ratio"] = credit_vals.values / age_safe
            created.append("fe_credit_to_age_ratio")

        # 3. Installment burden interaction (Installment Commitment * Duration)
        inst_col = (
            cols_lower.get("installment_commitment")
            or cols_lower.get("installment_rate")
        )
        if inst_col and duration_col:
            inst_vals = pd.to_numeric(X_out[inst_col], errors="coerce").fillna(1.0)
            duration_vals = pd.to_numeric(X_out[duration_col], errors="coerce").fillna(1.0)
            
            X_out["fe_cumulative_installment_burden"] = inst_vals.values * duration_vals.values
            created.append("fe_cumulative_installment_burden")

        # 4. Existing credits intensity per year of active tenure
        exist_col = cols_lower.get("existing_credits")
        if exist_col and duration_col:
            exist_vals = pd.to_numeric(X_out[exist_col], errors="coerce").fillna(1.0)
            duration_vals = pd.to_numeric(X_out[duration_col], errors="coerce").fillna(12.0)
            years_safe = np.maximum(duration_vals.values / 12.0, 0.5)

            X_out["fe_credit_velocity"] = exist_vals.values / years_safe
            created.append("fe_credit_velocity")

        self.created_features_ = created
        return X_out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return list of output feature names.

        Parameters
        ----------
        input_features : Optional[List[str]], default=None
            Input column names.

        Returns
        -------
        List[str]
            List of input plus created feature column names.
        """
        base = input_features if input_features is not None else self.feature_names_in_
        return list(base) + [f for f in self.created_features_ if f not in base]
