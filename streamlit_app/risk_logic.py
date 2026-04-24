"""Churn olasılığından risk segmentasyonu ve aksiyon önerisi üretir."""
from dataclasses import dataclass
from typing import List


@dataclass
class RiskAssessment:
    """Bir müşterinin risk değerlendirmesi."""
    level: str
    color: str
    emoji: str
    headline: str
    description: str
    actions: List[str]


def assess_risk(probability: float) -> RiskAssessment:
    """
    Churn olasılığına göre risk seviyesi ve aksiyon önerisi üretir.

    Eşikler churn management pratiklerine göre:
    - < 0.30: Düşük risk (routine engagement)
    - 0.30-0.55: Orta risk (proactive outreach)
    - 0.55-0.80: Yüksek risk (retention offer)
    - >= 0.80: Çok yüksek risk (urgent intervention)
    """
    if probability < 0.30:
        return RiskAssessment(
            level="Düşük",
            color="green",
            emoji="✅",
            headline="Sadık müşteri profili",
            description=(
                "Bu müşterinin yakın zamanda hizmeti bırakma ihtimali düşük. "
                "Mevcut ilişkiyi korumak ve upsell fırsatlarını değerlendirmek öncelikli olmalı."
            ),
            actions=[
                "Rutin memnuniyet anketi gönder",
                "Yeni hizmetler için cross-sell fırsatlarını değerlendir",
                "Sadakat programına davet et",
            ],
        )

    if probability < 0.55:
        return RiskAssessment(
            level="Orta",
            color="orange",
            emoji="⚠️",
            headline="İzlenmesi gereken müşteri",
            description=(
                "Churn riski mevcut ama kritik seviyede değil. "
                "Proaktif iletişim ve hizmet kalitesi iyileştirmeleri fark yaratabilir."
            ),
            actions=[
                "Kullanım verilerini incele, memnuniyetsizlik sinyali ara",
                "Kişiselleştirilmiş bir kampanya gönder",
                "Müşteri hizmetlerinden proaktif arama planla",
            ],
        )

    if probability < 0.80:
        return RiskAssessment(
            level="Yüksek",
            color="red",
            emoji="🚨",
            headline="Retention aksiyonu gerekli",
            description=(
                "Müşterinin yakın zamanda hizmeti bırakma olasılığı yüksek. "
                "Retention ekibi devreye girmeli, somut bir teklif sunulmalı."
            ),
            actions=[
                "İndirim veya paket yükseltme teklifi sun",
                "Uzun vadeli kontrat geçişi için özel teşvik tasarla",
                "Teknik destek ve hizmet kalitesi sorunlarını gözden geçir",
                "Dedicated retention uzmanı ata",
            ],
        )

    return RiskAssessment(
        level="Çok Yüksek",
        color="red",
        emoji="🔥",
        headline="Acil müdahale gerekli",
        description=(
            "Bu müşteri kısa vadede hizmeti bırakma sinyali veriyor. "
            "Standart retention süreçleri yetersiz kalabilir, "
            "üst düzey müdahale ve kişisel yaklaşım şart."
        ),
        actions=[
            "Üst düzey yönetici ile görüşme planla",
            "Agresif retention teklifi (ciddi indirim + bonus hizmet)",
            "Hizmet geçmişini detaylı incele, root-cause analizi yap",
            "24-48 saat içinde temas kur",
        ],
    )