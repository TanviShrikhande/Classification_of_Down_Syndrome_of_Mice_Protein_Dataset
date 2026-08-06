from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


DEFAULT_CLASSIFIERS = {
    "LinearSVM": SVC(kernel="linear"),
    "RadialSVM": SVC(kernel="rbf"),
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=20, random_state=0),
    "AdaBoost": AdaBoostClassifier(random_state=0),
    "DecisionTree": DecisionTreeClassifier(random_state=0),
    "KNeighbors": KNeighborsClassifier(),
}


def train_test_split_data(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, test_size: float = 0.25) -> tuple[Any, Any, Any, Any]:
    return train_test_split(X, y, test_size=test_size, random_state=42)


def evaluate_classifiers(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    results = []
    for name, model in DEFAULT_CLASSIFIERS.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append({"Classifier": name, "Accuracy": accuracy_score(y_test, preds)})
    return pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)


def train_random_forest(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, n_estimators: int = 150) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    model.fit(X, y)
    return model


def train_mlp(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> MLPClassifier:
    model = MLPClassifier(hidden_layer_sizes=(200, 200), max_iter=1000, activation="relu", solver="adam", random_state=1)
    model.fit(X, y)
    return model


def train_xgboost(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Install it via pip install xgboost") from exc

    model = XGBClassifier()
    model.fit(X, y)
    return model
