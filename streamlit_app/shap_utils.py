"""SHAP değerlerinin hesaplanması ve görselleştirilmesi."""
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model_pipeline.pkl"


def load_model_pipeline():
    """Eğitilmiş sklearn Pipeline'ı yükler (preprocessor + classifier)."""
    return joblib.load(MODEL_PATH)


def compute_shap_values(
    pipeline,
    customer_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Tek bir müşteri için SHAP değerlerini hesaplar.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed = preprocessor.transform(customer_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out().tolist()

    explainer = shap.TreeExplainer(classifier)
    shap_output = explainer.shap_values(transformed)

    if isinstance(shap_output, list):
        shap_values = shap_output[1][0]
    elif shap_output.ndim == 3:
        shap_values = shap_output[0, :, 1]
    else:
        shap_values = shap_output[0]

    return shap_values, transformed[0], feature_names


def plot_shap_waterfall(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list,
    base_value: float,
    top_n: int = 12,
) -> plt.Figure:
    """
    Tek müşteri için SHAP waterfall plotu üretir.
    """
    abs_shap = np.abs(shap_values)
    top_idx = np.argsort(abs_shap)[::-1][:top_n]

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_values[top_idx]]

    clean_names = []
    for name, val in zip(
        [feature_names[i] for i in top_idx],
        [feature_values[i] for i in top_idx],
    ):
        clean = name.replace("num__", "").replace("cat__", "")
        clean_names.append(f"{clean} = {val:.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_idx))
    ax.barh(y_pos, shap_values[top_idx], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP değeri (churn olasılığına katkı)")
    ax.set_title(f"En etkili {top_n} feature")
    plt.tight_layout()
    return fig


def get_expected_value(pipeline) -> float:
    """Model'in ortalama tahmin olasılığı (SHAP base value)."""
    classifier = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = ev[1] if len(np.atleast_1d(ev)) > 1 else ev[0]
    return float(ev)