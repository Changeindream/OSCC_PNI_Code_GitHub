"""Leakage-safe exploratory feature selection, fusion, and conventional classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFdr, SelectFromModel, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

META_COLUMNS = {"patient_id", "label", "split", "center", "t_stage"}


@dataclass
class BaselineResult:
    name: str
    estimator: BaseEstimator
    validation_auc: float
    predictions: pd.DataFrame


def merge_feature_blocks(
    blocks: dict[str, pd.DataFrame],
    *,
    id_column: str = "patient_id",
    label_column: str = "label",
) -> pd.DataFrame:
    """Inner-join feature blocks by patient ID with prefixed feature names."""
    if not blocks:
        raise ValueError("At least one feature block is required.")
    merged: pd.DataFrame | None = None
    reference_labels: pd.Series | None = None
    for block_name, frame in blocks.items():
        if id_column not in frame or label_column not in frame:
            raise ValueError(f"Block '{block_name}' lacks {id_column} or {label_column}.")
        if frame[id_column].duplicated().any():
            raise ValueError(f"Block '{block_name}' has duplicate patient IDs.")
        indexed_labels = frame.set_index(id_column)[label_column].sort_index()
        if reference_labels is None:
            reference_labels = indexed_labels
        else:
            common = reference_labels.index.intersection(indexed_labels.index)
            if not reference_labels.loc[common].equals(indexed_labels.loc[common]):
                raise ValueError(f"Label mismatch detected in block '{block_name}'.")

        feature_columns = [column for column in frame if column not in META_COLUMNS]
        renamed = frame[[id_column, label_column] + feature_columns].rename(
            columns={column: f"{block_name}__{column}" for column in feature_columns}
        )
        if merged is None:
            merged = renamed
        else:
            renamed = renamed.drop(columns=[label_column])
            merged = merged.merge(renamed, on=id_column, how="inner", validate="one_to_one")
    assert merged is not None
    return merged


def _split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    required = {"patient_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("Feature matrices require patient_id and label columns.")
    feature_columns = [column for column in frame if column not in META_COLUMNS]
    if not feature_columns:
        raise ValueError("No feature columns were found.")
    X = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians).fillna(0.0)
    y = pd.to_numeric(frame["label"], errors="raise").astype(int).to_numpy()
    return X, y, frame["patient_id"].astype(str)


def _classifier_spaces(seed: int) -> dict[str, tuple[BaseEstimator, dict[str, Any]]]:
    try:
        from skopt.space import Categorical, Integer, Real
    except ImportError as exc:
        raise ImportError("Install the 'baselines' extra for Bayesian model tuning.") from exc

    classifiers: dict[str, tuple[BaseEstimator, dict[str, Any]]] = {
        "SVM": (
            SVC(probability=True, random_state=seed),
            {
                "model__C": Real(1e-3, 1e3, prior="log-uniform"),
                "model__gamma": Real(1e-5, 1.0, prior="log-uniform"),
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "model__n_estimators": Integer(100, 800),
                "model__max_depth": Integer(2, 20),
                "model__max_features": Categorical(["sqrt", "log2", None]),
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": Integer(3, 25),
                "model__weights": Categorical(["uniform", "distance"]),
            },
        ),
        "SGD": (
            SGDClassifier(loss="log_loss", random_state=seed),
            {
                "model__alpha": Real(1e-6, 1e-2, prior="log-uniform"),
                "model__penalty": Categorical(["l1", "l2", "elasticnet"]),
            },
        ),
    }
    try:
        from xgboost import XGBClassifier

        classifiers["XGBoost"] = (
            XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=-1),
            {
                "model__n_estimators": Integer(100, 700),
                "model__max_depth": Integer(2, 8),
                "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            },
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        classifiers["LightGBM"] = (
            LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1),
            {
                "model__n_estimators": Integer(100, 700),
                "model__num_leaves": Integer(7, 63),
                "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            },
        )
    except ImportError:
        pass
    return classifiers


def _pipeline(model: BaseEstimator, seed: int) -> Pipeline:
    lasso = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=1.0,
        class_weight="balanced",
        max_iter=10_000,
        random_state=seed,
    )
    return Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("fdr", SelectFdr(score_func=f_classif, alpha=0.10)),
            ("scale", StandardScaler()),
            ("lasso", SelectFromModel(lasso, threshold=1e-8)),
            ("model", model),
        ]
    )


def fit_exploratory_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    seed: int = 42,
    iterations: int = 40,
    folds: int = 5,
) -> list[BaselineResult]:
    """Tune the six reported classifiers using training data only and score validation."""
    try:
        from skopt import BayesSearchCV
    except ImportError as exc:
        raise ImportError("Install the 'baselines' extra for Bayesian model tuning.") from exc

    X_train, y_train, _ = _split_xy(train)
    X_validation, y_validation, patient_ids = _split_xy(validation)
    X_validation = X_validation.reindex(columns=X_train.columns, fill_value=0.0)
    cross_validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: list[BaselineResult] = []

    for name, (model, search_space) in _classifier_spaces(seed).items():
        search = BayesSearchCV(
            _pipeline(model, seed),
            search_spaces=search_space,
            n_iter=iterations,
            scoring="roc_auc",
            cv=cross_validation,
            n_jobs=-1,
            random_state=seed,
            refit=True,
        )
        search.fit(X_train, y_train)
        probabilities = search.predict_proba(X_validation)[:, 1]
        predictions = pd.DataFrame(
            {
                "patient_id": patient_ids,
                "y_true": y_validation,
                "probability": probabilities,
                "prediction": (probabilities >= 0.5).astype(int),
                "model": name,
            }
        )
        results.append(
            BaselineResult(
                name=name,
                estimator=search.best_estimator_,
                validation_auc=float(roc_auc_score(y_validation, probabilities)),
                predictions=predictions,
            )
        )
    return sorted(results, key=lambda result: result.validation_auc, reverse=True)


def save_baseline_results(results: list[BaselineResult], output_dir: str | Path) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Install the 'baselines' extra to save fitted models.") from exc
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = []
    for result in results:
        safe_name = result.name.lower().replace(" ", "_")
        result.predictions.to_csv(
            destination / f"{safe_name}_validation_predictions.csv", index=False
        )
        joblib.dump(result.estimator, destination / f"{safe_name}.joblib")
        summary.append({"model": result.name, "validation_auc": result.validation_auc})
    pd.DataFrame(summary).to_csv(destination / "model_summary.csv", index=False)


def compare_feature_runs(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    patient_id: str,
) -> pd.DataFrame:
    """Return coordinate-wise paired features for the supplementary stability audit."""
    first_row = first.loc[first["patient_id"].astype(str) == str(patient_id)]
    second_row = second.loc[second["patient_id"].astype(str) == str(patient_id)]
    if len(first_row) != 1 or len(second_row) != 1:
        raise ValueError("The selected patient must occur exactly once in each run.")
    features = sorted(
        (set(first.columns) & set(second.columns)) - META_COLUMNS,
        key=lambda name: int(name.rsplit("_", 1)[-1])
        if name.rsplit("_", 1)[-1].isdigit()
        else name,
    )
    return pd.DataFrame(
        {
            "feature": features,
            "run_1": [float(first_row.iloc[0][feature]) for feature in features],
            "run_2": [float(second_row.iloc[0][feature]) for feature in features],
        }
    )


def lasso_ranked_features(
    training_frame: pd.DataFrame,
    *,
    top_k: int,
    seed: int = 42,
    folds: int = 10,
) -> pd.DataFrame:
    """Rank training-only features by absolute LASSO logistic coefficient.

    This helper supports the reported direct-fusion quotas (10 radiomics and 20
    pretrained deep features). It must be fitted independently within each training
    feature block; validation and test data are never used for ranking.
    """
    from sklearn.linear_model import LogisticRegressionCV

    if top_k < 1:
        raise ValueError("top_k must be positive.")
    X, y, _ = _split_xy(training_frame)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(X)
    selector = LogisticRegressionCV(
        Cs=20,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        penalty="l1",
        solver="saga",
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=20_000,
        random_state=seed,
        n_jobs=-1,
    )
    selector.fit(standardized, y)
    coefficients = selector.coef_.ravel()
    ranking = pd.DataFrame(
        {
            "feature": X.columns,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values(["absolute_coefficient", "feature"], ascending=[False, True])
    nonzero = ranking[ranking["absolute_coefficient"] > 0]
    selected = nonzero.head(top_k) if len(nonzero) >= top_k else ranking.head(top_k)
    return selected.reset_index(drop=True)
