"""API'ye istek atan helper modülü."""
import requests
from typing import Dict, Any


API_BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
HEALTH_ENDPOINT = f"{API_BASE_URL}/"


def check_api_health() -> bool:
    """API'nin ayakta olup olmadığını kontrol eder."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def predict_churn(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Müşteri verisini API'ye gönderir ve tahmini döndürür.

    Parameters
    ----------
    customer_data : dict
        schemas.CustomerData ile uyumlu 19 alanlı dict.

    Returns
    -------
    dict
        {prediction, prediction_label, churn_probability}

    Raises
    ------
    requests.exceptions.RequestException
        API erişilemezse veya 4xx/5xx dönerse.
    """
    response = requests.post(PREDICT_ENDPOINT, json=customer_data, timeout=10)
    response.raise_for_status()
    return response.json()