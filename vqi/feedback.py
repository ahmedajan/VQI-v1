"""VQI Feedback Template System.

Provides plain-language and expert-level feedback for VQI-S (signal quality)
and VQI-V (voice distinctiveness) scores.  Includes limiting-factor analysis,
category scoring, and full report export.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from vqi.engine import VQIResult

# ---------------------------------------------------------------------------
# Feedback templates  (category -> {issue, explain, fix, expert})
# ---------------------------------------------------------------------------

FEEDBACK_TEMPLATES_S = {
    "noise": {
        "issue": "Background noise detected",
        "explain": "Ambient noise reduces the clarity of voice features that speaker-verification systems rely on.",
        "fix": "Record in a quieter environment. Close doors and windows, turn off fans or AC, and keep the microphone close to your mouth.",
        "expert": "Signal-to-noise ratio and noise-floor features are below typical thresholds.",
    },
    "reverberation": {
        "issue": "Room echo / reverberation",
        "explain": "Reflected sound smears the speech signal and confuses recognition systems.",
        "fix": "Record in a smaller room with soft furnishings (carpet, curtains). Avoid large, empty spaces. Use a directional microphone if possible.",
        "expert": "Reverberation time (RT60) and clarity index (C50) indicate excessive room reflections.",
    },
    "spectral": {
        "issue": "Spectral quality concerns",
        "explain": "The frequency content of the recording deviates from what verification systems expect, possibly due to codec artifacts or microphone limitations.",
        "fix": "Use a higher-quality microphone and save in lossless format (WAV or FLAC). Avoid heavy compression or low-bitrate codecs.",
        "expert": "Spectral features (centroid, flux, cepstral coefficients) fall outside normative ranges.",
    },
    "dynamics": {
        "issue": "Volume instability or energy problems",
        "explain": "Large volume fluctuations or consistently low energy make it harder for systems to extract stable voice features.",
        "fix": "Maintain a steady distance from the microphone. Speak at a consistent volume. Adjust input gain to avoid clipping or overly quiet levels.",
        "expert": "Energy and dynamics features show abnormal variance or out-of-range values.",
    },
    "voice": {
        "issue": "Voice quality issues",
        "explain": "Voice characteristics (pitch stability, harmonic structure) differ from what systems expect for clean speech.",
        "fix": "Speak in your natural voice without whispering or shouting. Ensure no vocal strain. If the issue persists, it may reflect normal voice characteristics.",
        "expert": "HNR, pitch, jitter/shimmer, and glottal features are outside normal ranges.",
    },
    "temporal": {
        "issue": "Speech timing / pacing issues",
        "explain": "Unusual speech rate, long pauses, or very short utterances reduce the amount of usable voice data.",
        "fix": "Speak at a natural pace for at least 5-10 seconds. Avoid very long pauses between words.",
        "expert": "Speech rate, pause patterns, and VAD ratio indicate suboptimal temporal structure.",
    },
    "channel": {
        "issue": "Recording channel artifacts",
        "explain": "Technical issues like DC offset, power-line hum, or bandwidth limitations degrade signal quality.",
        "fix": "Check your recording equipment for ground loops (hum). Use a properly calibrated microphone with adequate bandwidth.",
        "expert": "Channel-specific features (DC offset, hum, bandwidth) indicate hardware/transmission artifacts.",
    },
}

FEEDBACK_TEMPLATES_V = {
    "cepstral": {
        "issue": "Low spectral uniqueness",
        "explain": "The cepstral (spectral envelope) characteristics of this voice sample are close to population average, making the voice less distinctive.",
        "fix": "Record a longer sample with varied sentence content. Use natural, conversational speech rather than reading in a monotone.",
        "expert": "MFCC, LPCC, and related cepstral features show low deviation from population means.",
    },
    "formant": {
        "issue": "Low formant distinctiveness",
        "explain": "Vowel formant patterns are close to average, which reduces voice uniqueness for verification.",
        "fix": "Speak naturally with varied vowel content. Conversational speech typically produces more distinctive formant patterns than careful reading.",
        "expert": "Formant frequencies (F1/F2/F3) and vocal tract length features are near population median.",
    },
    "prosodic": {
        "issue": "Low prosodic uniqueness",
        "explain": "Pitch patterns and speaking rhythm are close to average, providing fewer distinguishing cues.",
        "fix": "Use natural, expressive speech. Monotone reading produces less distinctive prosodic features.",
        "expert": "F0 statistics, speech rate, and rhythm features show low inter-speaker discrimination.",
    },
}


# ---------------------------------------------------------------------------
# Limiting-factor analysis
# ---------------------------------------------------------------------------

def find_limiting_factors(
    features_dict: Dict[str, float],
    importances_dict: Dict[str, float],
    percentiles: np.ndarray,
    percentile_names: List[str],
    categories: Dict[str, str],
    n_top: int = 5,
) -> List[dict]:
    """Identify the top features most limiting the VQI score.

    Ranking: ``importance * (1 - percentile_rank / 100)``
    High-importance features with low percentile ranks are the biggest problems.

    Returns list of dicts with keys:
        feature_name, value, percentile, category, importance, score
    """
    name_to_idx = {n: i for i, n in enumerate(percentile_names)}
    scored = []
    for fname, fval in features_dict.items():
        if fname not in name_to_idx or fname not in importances_dict:
            continue
        idx = name_to_idx[fname]
        pct_row = percentiles[idx]  # (101,)
        # Find percentile rank of this value
        pct_rank = int(np.searchsorted(pct_row, fval, side="right"))
        pct_rank = min(pct_rank, 100)
        imp = importances_dict[fname]
        # Score: important AND low-scoring features get high urgency
        urgency = imp * (1.0 - pct_rank / 100.0)
        scored.append({
            "feature_name": fname,
            "value": float(fval),
            "percentile": pct_rank,
            "category": categories.get(fname, "other"),
            "importance": float(imp),
            "score": float(urgency),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:n_top]


# ---------------------------------------------------------------------------
# Category scoring
# ---------------------------------------------------------------------------

def compute_category_scores(
    features_dict: Dict[str, float],
    percentiles: np.ndarray,
    percentile_names: List[str],
    categories: Dict[str, str],
) -> Dict[str, int]:
    """Compute a 0-100 score for each category based on mean percentile.

    Higher = better (feature values are at higher percentiles within the
    training distribution).
    """
    name_to_idx = {n: i for i, n in enumerate(percentile_names)}
    cat_pcts: Dict[str, list] = {}
    for fname, fval in features_dict.items():
        if fname not in name_to_idx:
            continue
        idx = name_to_idx[fname]
        pct_row = percentiles[idx]
        pct_rank = int(np.searchsorted(pct_row, fval, side="right"))
        pct_rank = min(pct_rank, 100)
        cat = categories.get(fname, "other")
        cat_pcts.setdefault(cat, []).append(pct_rank)
    result = {}
    for cat, pcts in cat_pcts.items():
        result[cat] = int(round(np.mean(pcts)))
    return result


# ---------------------------------------------------------------------------
# Plain-language feedback rendering
# ---------------------------------------------------------------------------

def render_plain_feedback(
    limiting_factors: List[dict],
    templates: dict,
    score: int,
    score_type: str = "S",
) -> str:
    """Render user-friendly plain-language feedback.

    Sections: "What's Good" (for high scores) and "What to Improve".
    """
    lines = []

    # Determine quality label
    if score >= 86:
        quality = "Excellent"
    elif score >= 71:
        quality = "Good"
    elif score >= 51:
        quality = "Fair"
    elif score >= 31:
        quality = "Below Average"
    else:
        quality = "Poor"

    type_label = "Signal Quality" if score_type == "S" else "Voice Distinctiveness"
    lines.append(f"{type_label}: {score}/100 ({quality})")
    lines.append("")

    if score >= 71:
        lines.append("What's Good:")
        if score_type == "S":
            lines.append("  Your recording has good overall signal quality for speaker verification.")
        else:
            lines.append("  Your voice shows distinctive characteristics that aid speaker verification.")
        if score >= 86:
            lines.append("  No significant issues detected.")
        lines.append("")

    # What to improve
    if limiting_factors:
        triggered_cats = set()
        for lf in limiting_factors:
            triggered_cats.add(lf["category"])

        improvement_items = []
        for cat in triggered_cats:
            tmpl = templates.get(cat)
            if tmpl:
                improvement_items.append(tmpl)

        if improvement_items:
            lines.append("What to Improve:")
            for tmpl in improvement_items:
                lines.append(f"  - {tmpl['issue']}")
                lines.append(f"    {tmpl['explain']}")
                lines.append(f"    Fix: {tmpl['fix']}")
                lines.append("")

    if not limiting_factors or score >= 86:
        if score >= 86:
            lines.append("No significant issues to address.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Expert feedback rendering
# ---------------------------------------------------------------------------

def render_expert_feedback(
    features_dict: Dict[str, float],
    percentiles: np.ndarray,
    percentile_names: List[str],
    categories: Dict[str, str],
    category_scores: Dict[str, int],
    limiting_factors: List[dict],
    score: int,
    score_type: str = "S",
) -> str:
    """Render expert-level diagnostic feedback with percentile context."""
    lines = []
    type_label = "VQI-S" if score_type == "S" else "VQI-V"
    lines.append(f"{type_label} Expert Diagnostics (Score: {score}/100)")
    lines.append("=" * 60)
    lines.append("")

    # Category breakdown table
    lines.append("Category Scores:")
    lines.append(f"  {'Category':<20} {'Score':>6}  {'Rating':<12}")
    lines.append(f"  {'-'*20} {'-'*6}  {'-'*12}")
    for cat in sorted(category_scores.keys()):
        cs = category_scores[cat]
        if cs >= 71:
            rating = "Good"
        elif cs >= 51:
            rating = "Fair"
        elif cs >= 31:
            rating = "Low"
        else:
            rating = "Very Low"
        lines.append(f"  {cat:<20} {cs:>5}/100  {rating:<12}")
    lines.append("")

    # Top limiting factors
    if limiting_factors:
        lines.append("Top Limiting Factors:")
        lines.append(f"  {'Feature':<30} {'Value':>10} {'Pct':>5} {'Category':<15} {'Importance':>10}")
        lines.append(f"  {'-'*30} {'-'*10} {'-'*5} {'-'*15} {'-'*10}")
        for lf in limiting_factors:
            lines.append(
                f"  {lf['feature_name']:<30} {lf['value']:>10.4f} "
                f"P{lf['percentile']:<4} {lf['category']:<15} {lf['importance']:>10.6f}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Overall assessment
# ---------------------------------------------------------------------------

_ASSESSMENT_MATRIX = {
    # (s_tier, v_tier) -> assessment text
    ("high", "high"): "Excellent overall. Both signal quality and voice distinctiveness are strong -- ideal for speaker verification.",
    ("high", "mid"): "Strong signal quality with moderate voice distinctiveness. The recording is clean, but the voice may be harder to distinguish from others.",
    ("high", "low"): "Clean recording, but low voice distinctiveness. The signal is excellent, but the voice profile may not stand out for verification.",
    ("mid", "high"): "Moderate signal quality with distinctive voice characteristics. Improving recording conditions would further boost performance.",
    ("mid", "mid"): "Moderate on both dimensions. There is room for improvement in both recording quality and voice distinctiveness.",
    ("mid", "low"): "Moderate signal quality and low voice distinctiveness. Focus on improving recording conditions first.",
    ("low", "high"): "Poor signal quality despite distinctive voice. The recording issues are masking an otherwise unique voice -- improve recording conditions.",
    ("low", "mid"): "Below-average signal quality and moderate distinctiveness. Recording conditions need significant improvement.",
    ("low", "low"): "Poor overall. Both signal quality and voice distinctiveness are low -- consider re-recording in a better environment.",
}


def _score_to_tier(score: int) -> str:
    if score >= 71:
        return "high"
    elif score >= 41:
        return "mid"
    else:
        return "low"


def generate_overall_assessment(score_s: int, score_v: int) -> str:
    """Generate a 1-2 sentence overall assessment combining S and V scores."""
    tier_s = _score_to_tier(score_s)
    tier_v = _score_to_tier(score_v)
    return _ASSESSMENT_MATRIX.get(
        (tier_s, tier_v),
        f"VQI-S: {score_s}/100, VQI-V: {score_v}/100.",
    )


# ---------------------------------------------------------------------------
# Export report
# ---------------------------------------------------------------------------

def generate_export_report(result: "VQIResult") -> str:
    """Generate a full text report for file export."""
    lines = []
    lines.append("=" * 70)
    lines.append("VQI - Voice Quality Index Report")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Scores
    lines.append("SCORES")
    lines.append("-" * 40)
    lines.append(f"  Signal Quality  (VQI-S): {result.score_s}/100")
    lines.append(f"  Voice Distinct. (VQI-V): {result.score_v}/100")
    lines.append("")

    # Overall assessment
    lines.append("OVERALL ASSESSMENT")
    lines.append("-" * 40)
    lines.append(f"  {result.overall_assessment}")
    lines.append("")

    # Audio info
    lines.append("AUDIO INFORMATION")
    lines.append("-" * 40)
    info = result.audio_info
    lines.append(f"  Duration:        {info.get('duration_s', 0):.1f}s")
    lines.append(f"  Speech duration: {info.get('speech_duration_s', 0):.1f}s")
    lines.append(f"  Speech ratio:    {info.get('speech_ratio', 0):.1%}")
    lines.append(f"  Sample rate:     {info.get('sample_rate', 16000)} Hz")
    if info.get("warnings"):
        lines.append(f"  Warnings:        {', '.join(info['warnings'])}")
    lines.append(f"  Processing time: {result.processing_time_ms:.0f} ms")
    lines.append("")

    # Plain feedback
    lines.append("SIGNAL QUALITY FEEDBACK")
    lines.append("-" * 40)
    lines.append(result.plain_feedback_s)
    lines.append("")

    lines.append("VOICE DISTINCTIVENESS FEEDBACK")
    lines.append("-" * 40)
    lines.append(result.plain_feedback_v)
    lines.append("")

    # Expert details
    lines.append("EXPERT DIAGNOSTICS")
    lines.append("-" * 40)
    lines.append(result.expert_feedback_s)
    lines.append("")
    lines.append(result.expert_feedback_v)
    lines.append("")

    # Category scores
    lines.append("CATEGORY SCORES (VQI-S)")
    lines.append("-" * 40)
    for cat, sc in sorted(result.category_scores_s.items()):
        lines.append(f"  {cat:<20} {sc:>3}/100")
    lines.append("")

    lines.append("CATEGORY SCORES (VQI-V)")
    lines.append("-" * 40)
    for cat, sc in sorted(result.category_scores_v.items()):
        lines.append(f"  {cat:<20} {sc:>3}/100")
    lines.append("")

    lines.append("=" * 70)
    lines.append("End of Report")
    return "\n".join(lines)
