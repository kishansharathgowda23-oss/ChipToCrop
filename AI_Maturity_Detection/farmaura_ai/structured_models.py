"""Optional structured-data baselines; requires reviewed, labelled tabular rows."""
from __future__ import annotations


def build_structured_classifier(name: str, random_state: int = 42):
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError("Install the optional scikit-learn package for structured models.") from exc

    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ))
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("Install optional `xgboost` to use this benchmark.") from exc
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                             eval_metric="mlogloss", random_state=random_state)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("Install optional `lightgbm` to use this benchmark.") from exc
        return LGBMClassifier(n_estimators=300, learning_rate=0.05,
                              class_weight="balanced", random_state=random_state)
    raise ValueError("Choose logistic_regression, random_forest, xgboost, or lightgbm.")
