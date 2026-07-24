"""
ExplainCrop — Fertilizer recommendation engine.

Building this in pieces. Piece 1: the knowledge base.

This is deliberately NOT the paper's novelty (we flagged this early on —
a rule-based lookup table is standard agronomic guidance, not a research
contribution by itself). The novelty is in piece 2/3: how the recommendation
RESPONDS to model confidence and severity uncertainty, not the base table
itself. So this piece should be treated as "known facts", sourced from
standard agri-extension guidance, not something to over-engineer.

Dosages are per-acre baselines for illustrative/demo purposes — NOT
agronomic advice to actually apply in the field without a real soil test.
This must be stated clearly in both the UI and the paper's limitations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FertilizerAdvice:
    fertilizer_name: str
    dosage_kg_per_acre: float
    application_method: str
    notes: str = ""


# Keyed by (crop, deficiency_class). Crops not listed for a given
# deficiency fall back to FALLBACK_ADVICE below (e.g. micronutrients on
# crops we didn't collect banana/coffee-specific dosages for).
FERTILIZER_KB = {
    ("rice", "nitrogen"): FertilizerAdvice(
        "Urea", 40, "Split into 2-3 doses across the growth cycle",
        "Avoid applying right before heavy rain — risk of runoff."),
    ("rice", "phosphorus"): FertilizerAdvice(
        "Single Super Phosphate (SSP)", 30, "Apply once at basal/sowing stage"),
    ("rice", "potassium"): FertilizerAdvice(
        "Muriate of Potash (MOP)", 25, "Split into 2 doses"),

    ("maize", "nitrogen"): FertilizerAdvice(
        "Urea", 50, "Split into 3 doses: basal, knee-high, tasseling"),
    ("maize", "phosphorus"): FertilizerAdvice(
        "DAP (Di-Ammonium Phosphate)", 35, "Apply at sowing"),
    ("maize", "potassium"): FertilizerAdvice(
        "Muriate of Potash (MOP)", 30, "Split into 2 doses"),

    ("tomato", "nitrogen"): FertilizerAdvice(
        "Calcium Ammonium Nitrate (CAN)", 25, "Split into weekly doses during fruiting"),
    ("tomato", "potassium"): FertilizerAdvice(
        "Sulphate of Potash (SOP)", 20, "Apply during flowering/fruit-set"),
    ("tomato", "magnesium"): FertilizerAdvice(
        "Magnesium Sulphate (Epsom salt)", 10, "Foliar spray, 2% solution, weekly"),
    ("tomato", "iron"): FertilizerAdvice(
        "Iron chelate (Fe-EDTA)", 5, "Foliar spray, biweekly"),

    ("wheat", "nitrogen"): FertilizerAdvice(
        "Urea", 45, "Split into 2 doses: basal + first irrigation"),
}

FALLBACK_ADVICE = FertilizerAdvice(
    "Consult local agri-extension officer",
    0, "n/a",
    "No dosage data available for this crop/nutrient combination in our "
    "knowledge base yet.")


def lookup_fertilizer(crop: str, deficiency_class: str) -> FertilizerAdvice:
    return FERTILIZER_KB.get((crop, deficiency_class), FALLBACK_ADVICE)


# ---------------------------------------------------------------------------
# Piece 2: confidence-aware scaling.
#
# This is the actual contribution, not the lookup table above. The idea:
# a model that is UNSURE should never recommend a full dose — that's how
# misdiagnosis turns into wasted money or crop damage. We combine two
# uncertainty signals that already exist elsewhere in the pipeline:
#
#   1. severity_variance  — from models.py's heteroscedastic severity head
#      (exp(severity_logvar)). High variance = model isn't confident about
#      how bad the deficiency actually is.
#   2. classification_confidence — softmax probability of the predicted
#      deficiency class. Low confidence = model isn't even sure WHICH
#      nutrient is missing.
#
# Both get combined into one 0-1 "trust score" that scales the recommended
# dosage down, and changes the message tone shown to the farmer.
# ---------------------------------------------------------------------------
import math


@dataclass
class ConfidenceAwareRecommendation:
    fertilizer_name: str
    recommended_dosage_kg_per_acre: float
    full_dosage_kg_per_acre: float
    application_method: str
    trust_score: float          # 0-1, higher = more confident
    tone: str                   # "confident" | "cautious" | "uncertain"
    message: str
    notes: str = ""


def compute_trust_score(classification_confidence: float,
                         severity_variance: float,
                         affected_area_ratio: float) -> float:
    """Combines three independent uncertainty signals into one 0-1 score.

    - classification_confidence: already 0-1 (softmax prob of predicted class)
    - severity_variance: unbounded >=0, so we squash it with exp(-x) —
      variance near 0 -> squashed value near 1 (confident); large variance
      -> squashed value near 0 (unsure)
    - affected_area_ratio: 0-1 from severity.py's HSV heuristic; used as a
      sanity cross-check — if Grad-CAM/color evidence shows almost no
      affected area but the model still claims a severe deficiency, that
      mismatch should also reduce trust
    """
    variance_term = math.exp(-severity_variance)
    consistency_term = 1.0 - abs(affected_area_ratio - severity_variance) if severity_variance <= 1 else 0.5

    trust = (0.5 * classification_confidence
             + 0.3 * variance_term
             + 0.2 * max(0.0, min(1.0, consistency_term)))
    return max(0.0, min(1.0, trust))


def recommend(crop: str, deficiency_class: str,
              classification_confidence: float,
              severity_variance: float,
              affected_area_ratio: float) -> ConfidenceAwareRecommendation:
    base = lookup_fertilizer(crop, deficiency_class)
    trust = compute_trust_score(classification_confidence, severity_variance, affected_area_ratio)

    if trust >= 0.7:
        tone = "confident"
        scale = 1.0
        message = (f"High confidence diagnosis. Apply the full recommended "
                    f"dose of {base.fertilizer_name}.")
    elif trust >= 0.4:
        tone = "cautious"
        scale = 0.5
        message = (f"Moderate confidence. Starting with a half-dose of "
                    f"{base.fertilizer_name} is safer — re-scan in 7-10 days "
                    f"before applying the rest.")
    else:
        tone = "uncertain"
        scale = 0.0
        message = (f"Low confidence diagnosis — we're not sure enough to "
                    f"recommend a dosage. Please get a soil test or consult "
                    f"your local agri-extension officer before applying "
                    f"{base.fertilizer_name}.")

    return ConfidenceAwareRecommendation(
        fertilizer_name=base.fertilizer_name,
        recommended_dosage_kg_per_acre=round(base.dosage_kg_per_acre * scale, 1),
        full_dosage_kg_per_acre=base.dosage_kg_per_acre,
        application_method=base.application_method,
        trust_score=round(trust, 2),
        tone=tone,
        message=message,
        notes=base.notes,
    )
