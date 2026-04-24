"""
Telco Customer Churn — Streamlit Arayüzü (Dark Theme + Rich Visuals)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import check_api_health, predict_churn
from risk_logic import assess_risk
from shap_utils import (
    compute_shap_values,
    get_expected_value,
    load_model_pipeline,
    plot_shap_waterfall,
)


# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="Churn Tahmin & Açıklanabilirlik",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS — dark theme üzerine ince dokunuşlar
st.markdown("""
<style>
    /* Metric cards için glow efekti */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1F2E 0%, #252B3D 100%);
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #2A3142;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #9BA3B4;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    /* Risk kartı */
    .risk-card {
        padding: 20px;
        border-radius: 12px;
        margin: 16px 0;
        border-left: 6px solid;
    }
    .risk-low { background: rgba(39, 174, 96, 0.15); border-left-color: #27AE60; }
    .risk-medium { background: rgba(243, 156, 18, 0.15); border-left-color: #F39C12; }
    .risk-high { background: rgba(231, 76, 60, 0.15); border-left-color: #E74C3C; }
    .risk-critical { background: rgba(192, 57, 43, 0.2); border-left-color: #C0392B; }
    /* Section header */
    .section-header {
        color: #FF4B6E;
        border-bottom: 2px solid #2A3142;
        padding-bottom: 8px;
        margin-top: 24px;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
    background-color: #1A1F2E;
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    color: #FAFAFA;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 1rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B6E;
        color: #FFFFFF !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #FFFFFF !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Matplotlib dark theme
# ============================================================================
import matplotlib.pyplot as plt
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#0E1117",
    "axes.edgecolor": "#2A3142",
    "axes.labelcolor": "#FAFAFA",
    "xtick.color": "#9BA3B4",
    "ytick.color": "#9BA3B4",
    "grid.color": "#2A3142",
    "text.color": "#FAFAFA",
})


# ============================================================================
# Feature tanımları — schemas.CustomerData ile birebir uyumlu
# ============================================================================
CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

SAMPLE_PROFILES = {
    "🔥 Yüksek Riskli (Yeni Fiber Müşterisi)": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 191.0,
    },
    "✅ Düşük Riskli (Sadık Müşteri)": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 65.0, "TotalCharges": 3900.0,
    },
    "⚠️ Orta Riskli (Kararsız)": {
        "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "tenure": 18, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 990.0,
    },
}


# ============================================================================
# Session state
# ============================================================================
def init_session_state():
    defaults = SAMPLE_PROFILES["⚠️ Orta Riskli (Kararsız)"].copy()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "last_input" not in st.session_state:
        st.session_state.last_input = None
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def apply_sample_profile(profile_name: str):
    profile = SAMPLE_PROFILES[profile_name]
    for key, value in profile.items():
        st.session_state[key] = value


# ============================================================================
# Sidebar
# ============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("# 📉 Churn Predictor")
        st.caption("Telco Customer Churn · Veri Bilimi Challenge")

        st.divider()

        # API health
        st.markdown("### 🟢 API Durumu")
        api_ok = check_api_health()
        if api_ok:
            st.success("API çalışıyor · `localhost:8000`")
        else:
            st.error("API erişilemez")
            st.code("docker start churn-api", language="bash")

        st.divider()

        # Örnek profiller
        st.markdown("### 🎯 Örnek Profiller")
        st.caption("Hızlı test için:")
        for profile_name in SAMPLE_PROFILES.keys():
            if st.button(profile_name, use_container_width=True, key=f"profile_{profile_name}"):
                apply_sample_profile(profile_name)
                st.rerun()

        st.divider()

        # Tahmin geçmişi sayacı
        history_count = len(st.session_state.get("prediction_history", []))
        st.markdown(f"### 📊 Bu Oturum")
        st.metric("Toplam Tahmin", history_count)

        if history_count > 0:
            if st.button("🗑️ Geçmişi Temizle", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()

        st.divider()

        with st.expander("ℹ️ Proje Hakkında"):
            st.markdown(
                "- **Model:** Random Forest\n"
                "- **Veri:** Telco Churn (7043)\n"
                "- **API:** FastAPI + Docker\n"
                "- **XAI:** SHAP TreeExplainer"
            )


# ============================================================================
# Tab 1: Tahmin
# ============================================================================
def render_prediction_tab():
    st.markdown("### 📝 Müşteri Bilgileri")
    st.caption(
        "Form alanlarını doldur, profil değerleri sekmeler arası korunur. "
        "Sol menüden örnek profil de yükleyebilirsin."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demografi**")
        st.selectbox("Cinsiyet", CATEGORICAL_OPTIONS["gender"], key="gender")
        st.selectbox(
            "Yaşlı müşteri (65+)", [0, 1], key="SeniorCitizen",
            format_func=lambda x: "Evet" if x == 1 else "Hayır",
        )
        st.selectbox("Partner var mı?", CATEGORICAL_OPTIONS["Partner"], key="Partner")
        st.selectbox("Bakmakla yükümlü", CATEGORICAL_OPTIONS["Dependents"], key="Dependents")

        st.markdown("**📅 Müşteri Geçmişi**")
        st.number_input("Tenure (ay)", min_value=0, max_value=100, step=1, key="tenure",
                        help="Müşterinin hizmet aldığı ay sayısı")

    with col2:
        st.markdown("**📞 Telefon & İnternet**")
        st.selectbox("Telefon hizmeti", CATEGORICAL_OPTIONS["PhoneService"], key="PhoneService")
        st.selectbox("Çoklu hat", CATEGORICAL_OPTIONS["MultipleLines"], key="MultipleLines")
        st.selectbox("İnternet servisi", CATEGORICAL_OPTIONS["InternetService"], key="InternetService")
        st.selectbox("Online güvenlik", CATEGORICAL_OPTIONS["OnlineSecurity"], key="OnlineSecurity")
        st.selectbox("Online yedekleme", CATEGORICAL_OPTIONS["OnlineBackup"], key="OnlineBackup")
        st.selectbox("Cihaz koruma", CATEGORICAL_OPTIONS["DeviceProtection"], key="DeviceProtection")
        st.selectbox("Teknik destek", CATEGORICAL_OPTIONS["TechSupport"], key="TechSupport")

    with col3:
        st.markdown("**🎬 Streaming & Sözleşme**")
        st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"], key="StreamingTV")
        st.selectbox("Streaming Movies", CATEGORICAL_OPTIONS["StreamingMovies"], key="StreamingMovies")
        st.selectbox("Sözleşme tipi", CATEGORICAL_OPTIONS["Contract"], key="Contract")
        st.selectbox("Kağıtsız fatura", CATEGORICAL_OPTIONS["PaperlessBilling"], key="PaperlessBilling")
        st.selectbox("Ödeme yöntemi", CATEGORICAL_OPTIONS["PaymentMethod"], key="PaymentMethod")

        st.markdown("**💰 Ücretlendirme**")
        st.number_input("Aylık ücret ($)", min_value=0.0, max_value=200.0, step=0.5,
                        key="MonthlyCharges")
        st.number_input("Toplam ücret ($)", min_value=0.0, max_value=10000.0, step=1.0,
                        key="TotalCharges")

    st.divider()

    if st.button("🔮 Tahmin Et", type="primary", use_container_width=True):
        customer_data = _collect_form_data()
        _run_prediction(customer_data)

    if st.session_state.last_prediction is not None:
        _render_prediction_result(st.session_state.last_prediction)


def _collect_form_data() -> dict:
    fields = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]
    return {field: st.session_state[field] for field in fields}


def _run_prediction(customer_data: dict):
    try:
        with st.spinner("Tahmin yapılıyor..."):
            result = predict_churn(customer_data)
        st.session_state.last_prediction = result
        st.session_state.last_input = customer_data
        # Tahmin geçmişine ekle
        st.session_state.prediction_history.append({
            "probability": result["churn_probability"],
            "label": result["prediction_label"],
            "tenure": customer_data["tenure"],
            "contract": customer_data["Contract"],
            "monthly_charges": customer_data["MonthlyCharges"],
        })
    except Exception as e:
        st.error(f"API'ye ulaşılamadı: {e}")


def _render_prediction_result(result: dict):
    probability = result["churn_probability"]
    label = result["prediction_label"]
    risk = assess_risk(probability)

    st.markdown("### 🎯 Tahmin Sonucu")

    # Üst metrikler
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tahmin", label, delta="Churn" if result["prediction"] == 1 else "Sadık",
                  delta_color="inverse" if result["prediction"] == 1 else "normal")
    with col2:
        st.metric("Churn Olasılığı", f"%{probability * 100:.1f}")
    with col3:
        st.metric("Risk Seviyesi", f"{risk.emoji} {risk.level}")
    with col4:
        confidence = max(probability, 1 - probability) * 100
        st.metric("Güven", f"%{confidence:.1f}")

    # Olasılık gauge
    _render_probability_gauge(probability)

    # Risk kartı
    risk_class = {"green": "risk-low", "orange": "risk-medium",
                  "red": "risk-high" if probability < 0.80 else "risk-critical"}[risk.color]

    st.markdown(f"""
    <div class="risk-card {risk_class}">
        <h3 style="margin-top:0;">{risk.emoji} {risk.headline}</h3>
        <p style="margin-bottom:12px;">{risk.description}</p>
        <b>Önerilen Aksiyonlar:</b>
        <ul>
            {''.join(f'<li>{action}</li>' for action in risk.actions)}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Müşteri profili özet tablosu
    _render_customer_summary_table(st.session_state.last_input)

    st.info("💡 Bu tahmine neden ulaşıldığını görmek için **'Açıklanabilirlik'** sekmesine geç.")


def _render_probability_gauge(probability: float):
    """Olasılığı yatay bir gauge olarak göster."""
    fig, ax = plt.subplots(figsize=(12, 1.5))

    # Renk zonları
    zones = [
        (0.0, 0.30, "#27AE60"),   # yeşil
        (0.30, 0.55, "#F39C12"),  # turuncu
        (0.55, 0.80, "#E67E22"),  # koyu turuncu
        (0.80, 1.00, "#C0392B"),  # kırmızı
    ]
    for start, end, color in zones:
        ax.barh(0, end - start, left=start, color=color, height=0.5, alpha=0.7)

    # İşaretleyici
    ax.axvline(x=probability, color="white", linewidth=3, ymin=0.1, ymax=0.9)
    ax.scatter([probability], [0], color="white", s=200, zorder=5,
               edgecolor="#FF4B6E", linewidth=3)

    # Zon etiketleri
    zone_labels = [("Düşük", 0.15), ("Orta", 0.425), ("Yüksek", 0.675), ("Çok Yüksek", 0.90)]
    for label, pos in zone_labels:
        ax.text(pos, -0.45, label, ha="center", va="top", fontsize=9, color="#9BA3B4")

    ax.text(probability, 0.5, f"%{probability*100:.1f}",
            ha="center", va="bottom", fontsize=14, fontweight="bold", color="white")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, 0.9)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("Churn Olasılığı", color="#FAFAFA", pad=10, loc="left", fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_customer_summary_table(customer_data: dict):
    """Müşteri profilini özet tablo olarak göster."""
    with st.expander("📋 Müşteri Profili Özeti", expanded=False):
        # İki grup halinde göster
        group1 = {
            "Cinsiyet": customer_data["gender"],
            "Yaşlı Müşteri": "Evet" if customer_data["SeniorCitizen"] == 1 else "Hayır",
            "Partner": customer_data["Partner"],
            "Bakmakla Yükümlü": customer_data["Dependents"],
            "Tenure (ay)": customer_data["tenure"],
            "Sözleşme": customer_data["Contract"],
            "Ödeme Yöntemi": customer_data["PaymentMethod"],
            "Kağıtsız Fatura": customer_data["PaperlessBilling"],
        }
        group2 = {
            "Telefon Hizmeti": customer_data["PhoneService"],
            "İnternet Servisi": customer_data["InternetService"],
            "Online Güvenlik": customer_data["OnlineSecurity"],
            "Online Yedekleme": customer_data["OnlineBackup"],
            "Cihaz Koruma": customer_data["DeviceProtection"],
            "Teknik Destek": customer_data["TechSupport"],
            "Streaming TV": customer_data["StreamingTV"],
            "Streaming Movies": customer_data["StreamingMovies"],
        }

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Demografi & Sözleşme**")
            df1 = pd.DataFrame(list(group1.items()), columns=["Alan", "Değer"])
            st.dataframe(df1, hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**Hizmetler**")
            df2 = pd.DataFrame(list(group2.items()), columns=["Alan", "Değer"])
            st.dataframe(df2, hide_index=True, use_container_width=True)

        st.markdown("**Ücretlendirme**")
        pricing_df = pd.DataFrame([{
            "Aylık Ücret ($)": f"{customer_data['MonthlyCharges']:.2f}",
            "Toplam Ücret ($)": f"{customer_data['TotalCharges']:.2f}",
            "Ortalama Aylık (Toplam/Tenure)": (
                f"{customer_data['TotalCharges'] / customer_data['tenure']:.2f}"
                if customer_data['tenure'] > 0 else "N/A"
            ),
        }])
        st.dataframe(pricing_df, hide_index=True, use_container_width=True)


# ============================================================================
# Tab 2: SHAP
# ============================================================================
def render_explainability_tab():
    st.markdown("### 🔍 SHAP Açıklanabilirlik Analizi")
    st.caption(
        "Son yapılan tahmin için hangi feature'ların churn olasılığını nasıl etkilediğini gösterir."
    )

    if st.session_state.last_prediction is None:
        st.warning("Henüz tahmin yapılmadı. Önce **Tahmin** sekmesinden bir tahmin üret.")
        return

    last_input = st.session_state.last_input
    last_pred = st.session_state.last_prediction

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tahmin", last_pred["prediction_label"])
    with col2:
        st.metric("Churn Olasılığı", f"%{last_pred['churn_probability'] * 100:.1f}")

    try:
        with st.spinner("SHAP değerleri hesaplanıyor..."):
            pipeline = _get_cached_pipeline()
            customer_df = pd.DataFrame([last_input])
            shap_values, feature_values, feature_names = compute_shap_values(pipeline, customer_df)
            base_value = get_expected_value(pipeline)
    except Exception as e:
        st.error(f"SHAP hesaplanırken hata oluştu: {e}")
        return

    with col3:
        st.metric("Base Value", f"%{base_value * 100:.1f}",
                  help="Model'in ortalama tahmin olasılığı. SHAP değerleri bunun üstüne eklenir.")

    st.divider()

    # Waterfall plot
    st.markdown("#### Feature Katkıları (Waterfall)")
    fig = plot_shap_waterfall(shap_values, feature_values, feature_names, base_value, top_n=12)
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div style="padding:12px;background-color:rgba(231,76,60,0.15);border-radius:8px;'
            'border-left:4px solid #E74C3C;">'
            "🔴 <b>Kırmızı</b>: Churn olasılığını <b>artırıyor</b></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div style="padding:12px;background-color:rgba(52,152,219,0.15);border-radius:8px;'
            'border-left:4px solid #3498DB;">'
            "🔵 <b>Mavi</b>: Churn olasılığını <b>azaltıyor</b></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Feature katkıları tablosu + top 3 kartı
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 📊 Tüm Feature Katkıları")
        shap_df = pd.DataFrame({
            "Feature": [n.replace("num__", "").replace("cat__", "") for n in feature_names],
            "Değer": feature_values,
            "SHAP Katkısı": shap_values,
            "Mutlak Etki": np.abs(shap_values),
        }).sort_values("Mutlak Etki", ascending=False).head(15).reset_index(drop=True)

        shap_df["Yön"] = shap_df["SHAP Katkısı"].apply(lambda x: "⬆️ Artırıyor" if x > 0 else "⬇️ Azaltıyor")
        display_df = shap_df[["Feature", "Değer", "SHAP Katkısı", "Yön"]].copy()
        display_df["Değer"] = display_df["Değer"].round(3)
        display_df["SHAP Katkısı"] = display_df["SHAP Katkısı"].round(4)

        st.dataframe(display_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### 🏆 En Etkili 3 Feature")
        top3_idx = np.argsort(np.abs(shap_values))[::-1][:3]
        for rank, idx in enumerate(top3_idx, start=1):
            name = feature_names[idx].replace("num__", "").replace("cat__", "")
            value = feature_values[idx]
            shap_val = shap_values[idx]
            color = "#E74C3C" if shap_val > 0 else "#3498DB"
            direction = "Churn'ü artırıyor" if shap_val > 0 else "Churn'ü azaltıyor"

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1A1F2E 0%, #252B3D 100%);
                        padding: 16px; border-radius: 10px; margin-bottom: 12px;
                        border-left: 4px solid {color};">
                <div style="color: #9BA3B4; font-size: 0.85rem;">#{rank}</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #FAFAFA;">{name}</div>
                <div style="color: #9BA3B4; font-size: 0.9rem;">Değer: {value:.3f}</div>
                <div style="color: {color}; font-weight: 600; margin-top: 4px;">
                    {direction}<br>SHAP: {shap_val:+.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)


@st.cache_resource
def _get_cached_pipeline():
    return load_model_pipeline()


# ============================================================================
# Tab 3: Hakkında + Geçmiş grafikleri
# ============================================================================
def render_about_tab():
    # Tahmin geçmişi grafikleri
    history = st.session_state.get("prediction_history", [])
    if len(history) > 0:
        st.markdown("### 📈 Oturum İstatistikleri")
        st.caption(f"Bu oturumda {len(history)} tahmin yapıldı.")

        hist_df = pd.DataFrame(history)

        col1, col2 = st.columns(2)

        with col1:
            # Olasılık dağılımı
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(hist_df["probability"], bins=10, color="#FF4B6E",
                    edgecolor="#FAFAFA", alpha=0.8)
            ax.axvline(x=0.5, color="white", linestyle="--", alpha=0.5, label="Karar sınırı")
            ax.set_xlabel("Churn Olasılığı")
            ax.set_ylabel("Frekans")
            ax.set_title("Tahmin Olasılığı Dağılımı")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            # Churn vs Sadık
            counts = hist_df["label"].value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#E74C3C" if l == "Churn" else "#27AE60" for l in counts.index]
            bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="#FAFAFA")
            ax.set_ylabel("Tahmin Sayısı")
            ax.set_title("Churn vs Sadık Dağılımı")
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', color="white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Son tahminler tablosu
        with st.expander("📋 Son Tahminler Tablosu"):
            display = hist_df.tail(10).copy()
            display["probability"] = display["probability"].apply(lambda x: f"%{x*100:.1f}")
            display["monthly_charges"] = display["monthly_charges"].apply(lambda x: f"${x:.2f}")
            display.columns = ["Olasılık", "Etiket", "Tenure", "Sözleşme", "Aylık Ücret"]
            st.dataframe(display.iloc[::-1], hide_index=True, use_container_width=True)

        st.divider()

    st.markdown("### 🎯 Proje Hakkında")
    st.markdown("""
    Telco müşterilerinin hizmeti bırakıp bırakmayacağını tahmin eden ML sistemi.
    Model FastAPI ile servis edilir, bu Streamlit arayüzü üzerinden tüketilir ve
    SHAP açıklanabilirlik katmanıyla zenginleştirilir.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 📦 Bileşenler

        - **API** (`app/`) — FastAPI servisi, `/predict` endpoint
        - **Model** (`model/`) — Eğitim scripti ve deneme dosyaları
        - **Arayüz** (`streamlit_app/`) — Bu Streamlit uygulaması
        - **Docker** — API'yi containerize eder

        #### 🔧 Teknolojiler

        Python 3.11 · FastAPI · Pydantic · scikit-learn · SHAP · Streamlit · Docker · Matplotlib
        """)

    with col2:
        st.markdown("""
        #### 📊 Risk Segmentasyonu

        | Seviye | Olasılık | Aksiyon |
        |--------|----------|---------|
        | ✅ Düşük | < %30 | Upsell, sadakat |
        | ⚠️ Orta | %30-55 | Proaktif izleme |
        | 🚨 Yüksek | %55-80 | Retention teklifi |
        | 🔥 Çok Yüksek | ≥ %80 | Acil müdahale |

        #### 🧪 Model

        Random Forest Classifier · class_weight balanced ·
        500 estimators · max_depth 12
        """)

    st.markdown("#### 🏗️ Sistem Mimarisi")
    st.code("""
┌─────────────────┐      HTTP       ┌──────────────────┐
│  Streamlit UI   │ ─────────────▶ │  FastAPI Service │
│  (localhost)    │ ◀───────────── │  (Docker :8000)  │
└────────┬────────┘    JSON         └──────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────┐                 ┌──────────────────┐
│ SHAP Explainer  │                 │ RandomForest +   │
│ (TreeExplainer) │                 │ sklearn Pipeline │
└─────────────────┘                 └──────────────────┘
    """, language="text")


# ============================================================================
# Main
# ============================================================================
def main():
    init_session_state()
    render_sidebar()

    st.markdown("# 📉 Telco Customer Churn Tahmini")
    st.caption(
        "Müşteri profil bilgisi gir, churn tahminini gör, SHAP ile nedenini anla, "
        "önerilen aksiyonları uygula."
    )

    tab1, tab2, tab3 = st.tabs(["🔮 Tahmin", "🔍 Açıklanabilirlik", "ℹ️ Hakkında"])

    with tab1:
        render_prediction_tab()
    with tab2:
        render_explainability_tab()
    with tab3:
        render_about_tab()


if __name__ == "__main__":
    main()