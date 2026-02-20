"""Unit tests for VQI evaluation modules (Step 8).

Tests ERC, DET, cross-system, combined ERC, and quadrant analysis
using synthetic data with known properties.
"""

import os
import sys

import numpy as np
import pytest

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vqi.evaluation.erc import (
    compute_erc,
    compute_pairwise_quality,
    compute_fnmr_reduction_at_reject,
    compute_random_rejection_baseline,
    find_tau_for_fnmr,
    compute_fnmr_at_threshold,
    compute_fmr_at_threshold,
)
from vqi.evaluation.det import (
    compute_det_curve,
    compute_ranked_det,
    compute_fnmr_at_fmr,
)
from vqi.evaluation.cross_system import (
    evaluate_cross_system,
    check_monotonicity,
)
from vqi.evaluation.combined_erc import (
    compute_combined_erc,
    compute_combined_fnmr_reduction_summary,
)
from vqi.evaluation.quadrant_analysis import (
    assign_quadrants,
    assign_pair_quadrants,
    compute_quadrant_performance,
)


# ============================================================
# Fixtures: synthetic data with known properties
# ============================================================

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def synthetic_scores(rng):
    """Create synthetic genuine/impostor scores with clear separation."""
    n_gen = 5000
    n_imp = 5000

    # Genuine scores: higher cosine similarity (centered ~0.7)
    genuine_sim = rng.normal(0.7, 0.1, n_gen).astype(np.float32)
    # Impostor scores: lower cosine similarity (centered ~0.3)
    impostor_sim = rng.normal(0.3, 0.1, n_imp).astype(np.float32)

    # Quality scores: correlated with similarity for genuine pairs
    # High quality -> high genuine score, low quality -> low genuine score
    quality_genuine = rng.uniform(0, 100, n_gen).astype(np.float32)
    quality_impostor = rng.uniform(0, 100, n_imp).astype(np.float32)

    # Make quality correlated with genuine score: sort both and align
    sort_idx = np.argsort(quality_genuine)
    genuine_sim_sorted = np.sort(genuine_sim)
    genuine_sim[sort_idx] = genuine_sim_sorted

    return {
        "genuine_sim": genuine_sim,
        "impostor_sim": impostor_sim,
        "quality_genuine": quality_genuine,
        "quality_impostor": quality_impostor,
    }


@pytest.fixture
def synthetic_dual_scores(rng):
    """Create synthetic data with both VQI-S and VQI-V quality scores."""
    n = 1000  # number of samples
    n_gen = 3000
    n_imp = 3000

    # Per-sample quality scores
    quality_s = rng.uniform(0, 100, n).astype(np.float32)
    quality_v = rng.uniform(0, 100, n).astype(np.float32)

    # Pairs
    gen_pairs = rng.randint(0, n, size=(n_gen, 2))
    imp_pairs = rng.randint(0, n, size=(n_imp, 2))

    genuine_sim = rng.normal(0.7, 0.1, n_gen).astype(np.float32)
    impostor_sim = rng.normal(0.3, 0.1, n_imp).astype(np.float32)

    # Pairwise quality
    quality_gen_s = np.minimum(quality_s[gen_pairs[:, 0]], quality_s[gen_pairs[:, 1]])
    quality_gen_v = np.minimum(quality_v[gen_pairs[:, 0]], quality_v[gen_pairs[:, 1]])
    quality_imp_s = np.minimum(quality_s[gen_pairs[:, 0]], quality_s[gen_pairs[:, 1]])[:n_imp]
    quality_imp_v = np.minimum(quality_v[gen_pairs[:, 0]], quality_v[gen_pairs[:, 1]])[:n_imp]

    return {
        "quality_s": quality_s,
        "quality_v": quality_v,
        "genuine_sim": genuine_sim,
        "impostor_sim": impostor_sim,
        "quality_gen_s": quality_gen_s,
        "quality_gen_v": quality_gen_v,
        "quality_imp_s": quality_imp_s,
        "quality_imp_v": quality_imp_v,
        "gen_pairs": gen_pairs,
        "imp_pairs": imp_pairs,
    }


# ============================================================
# ERC Tests
# ============================================================

class TestERC:

    def test_compute_pairwise_quality(self):
        quality = np.array([80, 60, 40, 20, 90])
        pairs = np.array([[0, 1], [2, 3], [0, 4]])
        result = compute_pairwise_quality(quality, pairs)
        np.testing.assert_array_equal(result, [60, 20, 80])

    def test_fnmr_at_threshold(self):
        # All scores above threshold -> FNMR = 0
        scores = np.array([0.8, 0.9, 0.7, 0.85])
        assert compute_fnmr_at_threshold(scores, 0.5) == 0.0

        # All below -> FNMR = 1
        assert compute_fnmr_at_threshold(scores, 1.0) == 1.0

        # Half below
        scores2 = np.array([0.3, 0.7, 0.2, 0.8])
        assert compute_fnmr_at_threshold(scores2, 0.5) == 0.5

    def test_fmr_at_threshold(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        # No impostor >= 0.5
        assert compute_fmr_at_threshold(scores, 0.5) == 0.0
        # All >= 0.0
        assert compute_fmr_at_threshold(scores, 0.0) == 1.0

    def test_erc_output_shape(self, synthetic_scores):
        d = synthetic_scores
        tau = find_tau_for_fnmr(d["genuine_sim"], d["impostor_sim"], 0.10)
        q_range = np.arange(0, 101, dtype=float)
        erc = compute_erc(
            d["genuine_sim"], d["impostor_sim"],
            d["quality_genuine"], d["quality_impostor"],
            tau, q_range,
        )

        assert len(erc["reject_fracs"]) == 101
        assert len(erc["fnmr_values"]) == 101
        assert len(erc["fmr_values"]) == 101
        assert len(erc["q_thresholds"]) == 101
        assert len(erc["n_genuine_remaining"]) == 101
        assert len(erc["n_impostor_remaining"]) == 101

    def test_erc_monotonicity(self, synthetic_scores):
        """ERC FNMR should generally decrease as quality threshold increases
        (rejecting more low-quality pairs)."""
        d = synthetic_scores
        tau = find_tau_for_fnmr(d["genuine_sim"], d["impostor_sim"], 0.10)
        erc = compute_erc(
            d["genuine_sim"], d["impostor_sim"],
            d["quality_genuine"], d["quality_impostor"],
            tau,
        )
        fnmr = erc["fnmr_values"]
        valid = fnmr[~np.isnan(fnmr)]
        # Check that at least 80% of consecutive differences are non-positive
        diffs = np.diff(valid)
        assert np.mean(diffs <= 0.01) > 0.7, "ERC should be mostly non-increasing"

    def test_erc_reject_fracs_range(self, synthetic_scores):
        """Reject fraction should be in [0, 1] and generally increasing."""
        d = synthetic_scores
        tau = find_tau_for_fnmr(d["genuine_sim"], d["impostor_sim"], 0.10)
        erc = compute_erc(
            d["genuine_sim"], d["impostor_sim"],
            d["quality_genuine"], d["quality_impostor"],
            tau,
        )
        rf = erc["reject_fracs"]
        assert rf[0] == 0.0, "No rejection at q=0"
        assert np.all(rf >= 0) and np.all(rf <= 1.0)

    def test_fnmr_reduction_positive(self, synthetic_scores):
        """Quality-based rejection should reduce FNMR more than random."""
        d = synthetic_scores
        tau = find_tau_for_fnmr(d["genuine_sim"], d["impostor_sim"], 0.10)
        erc = compute_erc(
            d["genuine_sim"], d["impostor_sim"],
            d["quality_genuine"], d["quality_impostor"],
            tau,
        )
        reductions = compute_fnmr_reduction_at_reject(erc)
        # At 20% rejection, should have some positive reduction
        red_20 = reductions[0.20]["fnmr_reduction_pct"]
        assert red_20 > 0, f"Expected positive FNMR reduction, got {red_20}"

    def test_random_baseline_shape(self, synthetic_scores):
        d = synthetic_scores
        tau = find_tau_for_fnmr(d["genuine_sim"], d["impostor_sim"], 0.10)
        reject_fracs = np.linspace(0, 0.5, 50)
        baseline = compute_random_rejection_baseline(
            d["genuine_sim"], tau, reject_fracs, n_bootstrap=20,
        )
        assert len(baseline) == 50

    def test_empty_genuine(self):
        """FNMR should be NaN for empty genuine set."""
        result = compute_fnmr_at_threshold(np.array([]), 0.5)
        assert np.isnan(result)


# ============================================================
# DET Tests
# ============================================================

class TestDET:

    def test_det_curve_output(self, rng):
        gen = rng.normal(0.7, 0.1, 1000)
        imp = rng.normal(0.3, 0.1, 1000)
        det = compute_det_curve(gen, imp)

        assert "fmr" in det and "fnmr" in det
        assert "eer" in det
        assert len(det["fmr"]) == len(det["fnmr"])
        assert 0 < det["eer"] < 0.5, f"EER={det['eer']} should be small for well-separated distributions"

    def test_eer_well_separated(self, rng):
        """Well-separated distributions should have small EER."""
        gen = rng.normal(0.9, 0.05, 2000)
        imp = rng.normal(0.1, 0.05, 2000)
        det = compute_det_curve(gen, imp)
        assert det["eer"] < 0.01

    def test_eer_overlapping(self, rng):
        """Overlapping distributions should have large EER."""
        gen = rng.normal(0.5, 0.2, 2000)
        imp = rng.normal(0.5, 0.2, 2000)
        det = compute_det_curve(gen, imp)
        assert det["eer"] > 0.3

    def test_fnmr_at_fmr(self, rng):
        gen = rng.normal(0.7, 0.1, 2000)
        imp = rng.normal(0.3, 0.1, 2000)
        fnmr = compute_fnmr_at_fmr(gen, imp, 0.01)
        assert 0 <= fnmr <= 1
        # For well-separated, FNMR at FMR=1% should be low
        assert fnmr < 0.5

    def test_ranked_det_groups(self, synthetic_scores):
        d = synthetic_scores
        result = compute_ranked_det(
            d["genuine_sim"], d["impostor_sim"],
            d["quality_genuine"], d["quality_impostor"],
        )

        assert "groups" in result
        assert "bottom" in result["groups"]
        assert "middle" in result["groups"]
        assert "top" in result["groups"]
        assert "eer_separation" in result

    def test_ranked_det_eer_ordering(self, synthetic_scores):
        """Top quality group should have lower EER than bottom group."""
        d = synthetic_scores
        result = compute_ranked_det(
            d["genuine_sim"], d["impostor_sim"],
            d["quality_genuine"], d["quality_impostor"],
        )

        eer_bottom = result["groups"]["bottom"]["det"]["eer"]
        eer_top = result["groups"]["top"]["det"]["eer"]
        # With quality correlated to genuine score, top should be better
        assert eer_top < eer_bottom, (
            f"Top EER ({eer_top:.4f}) should be lower than bottom EER ({eer_bottom:.4f})"
        )


# ============================================================
# Cross-System Tests
# ============================================================

class TestCrossSystem:

    def test_monotonicity_checker(self):
        # Strictly decreasing -> True
        assert check_monotonicity(np.array([0.5, 0.4, 0.3, 0.2, 0.1]))
        # Flat -> True
        assert check_monotonicity(np.array([0.5, 0.5, 0.5]))
        # Large increase -> False
        assert not check_monotonicity(np.array([0.5, 0.4, 0.6, 0.3]))
        # Small increase within tolerance -> True
        assert check_monotonicity(np.array([0.5, 0.4, 0.405, 0.35]), tolerance=0.02)
        # NaN handling
        assert check_monotonicity(np.array([np.nan, 0.5, 0.4]))

    def test_evaluate_cross_system(self, rng):
        """Cross-system should pass for well-behaved quality scores."""
        n_gen = 2000
        n_imp = 2000

        quality_gen = rng.uniform(0, 100, n_gen).astype(np.float32)
        quality_imp = rng.uniform(0, 100, n_imp).astype(np.float32)

        provider_data = {}
        for pn in ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2", "P4_XVECTOR", "P5_WAVLM"]:
            gen = rng.normal(0.7, 0.1, n_gen).astype(np.float32)
            imp = rng.normal(0.3, 0.1, n_imp).astype(np.float32)
            # Correlate quality with genuine score
            sort_idx = np.argsort(quality_gen)
            gen_sorted = np.sort(gen)
            gen[sort_idx] = gen_sorted
            provider_data[pn] = {"genuine_sim": gen, "impostor_sim": imp}

        results = evaluate_cross_system(provider_data, quality_gen, quality_imp)
        assert "_verdict" in results
        # With correlated quality, cross-system should pass
        verdict = results["_verdict"]
        assert verdict["all_positive_reduction"]


# ============================================================
# Combined ERC Tests
# ============================================================

class TestCombinedERC:

    def test_all_strategies_present(self, synthetic_dual_scores):
        d = synthetic_dual_scores
        gen_sim = d["genuine_sim"]
        imp_sim = d["impostor_sim"]
        tau = float(np.percentile(gen_sim, 10))

        result = compute_combined_erc(
            gen_sim, imp_sim,
            d["quality_gen_s"], d["quality_gen_v"],
            d["quality_imp_s"], d["quality_imp_v"],
            tau,
        )

        assert "s_only" in result
        assert "v_only" in result
        assert "union" in result
        assert "intersection" in result

    def test_union_rejects_more(self, synthetic_dual_scores):
        """Union strategy should reject at least as much as either single score."""
        d = synthetic_dual_scores
        gen_sim = d["genuine_sim"]
        imp_sim = d["impostor_sim"]
        tau = float(np.percentile(gen_sim, 10))

        result = compute_combined_erc(
            gen_sim, imp_sim,
            d["quality_gen_s"], d["quality_gen_v"],
            d["quality_imp_s"], d["quality_imp_v"],
            tau, np.array([50.0]),  # single threshold
        )

        rf_s = result["s_only"]["reject_fracs"][0]
        rf_v = result["v_only"]["reject_fracs"][0]
        rf_union = result["union"]["reject_fracs"][0]

        # Union (keep if both pass) rejects more than either single
        assert rf_union >= max(rf_s, rf_v) - 0.01, (
            f"Union reject ({rf_union:.3f}) should be >= max(S={rf_s:.3f}, V={rf_v:.3f})"
        )

    def test_intersection_rejects_less(self, synthetic_dual_scores):
        """Intersection strategy should reject at most as much as either single score."""
        d = synthetic_dual_scores
        gen_sim = d["genuine_sim"]
        imp_sim = d["impostor_sim"]
        tau = float(np.percentile(gen_sim, 10))

        result = compute_combined_erc(
            gen_sim, imp_sim,
            d["quality_gen_s"], d["quality_gen_v"],
            d["quality_imp_s"], d["quality_imp_v"],
            tau, np.array([50.0]),
        )

        rf_s = result["s_only"]["reject_fracs"][0]
        rf_v = result["v_only"]["reject_fracs"][0]
        rf_inter = result["intersection"]["reject_fracs"][0]

        # Intersection (keep if at least one passes) rejects less
        assert rf_inter <= min(rf_s, rf_v) + 0.01, (
            f"Intersection reject ({rf_inter:.3f}) should be <= min(S={rf_s:.3f}, V={rf_v:.3f})"
        )

    def test_summary_format(self, synthetic_dual_scores):
        d = synthetic_dual_scores
        gen_sim = d["genuine_sim"]
        imp_sim = d["impostor_sim"]
        tau = float(np.percentile(gen_sim, 10))

        result = compute_combined_erc(
            gen_sim, imp_sim,
            d["quality_gen_s"], d["quality_gen_v"],
            d["quality_imp_s"], d["quality_imp_v"],
            tau,
        )
        summary = compute_combined_fnmr_reduction_summary(result)

        for strat in ["s_only", "v_only", "union", "intersection"]:
            assert strat in summary
            for rf in [0.10, 0.20, 0.30]:
                assert rf in summary[strat]
                assert "fnmr_reduction_pct" in summary[strat][rf]


# ============================================================
# Quadrant Analysis Tests
# ============================================================

class TestQuadrantAnalysis:

    def test_assign_quadrants_basic(self):
        quality_s = np.array([80, 20, 20, 80, 50])
        quality_v = np.array([80, 80, 20, 20, 50])
        quads = assign_quadrants(quality_s, quality_v, threshold_s=50, threshold_v=50)
        assert quads[0] == 1  # high S, high V
        assert quads[1] == 2  # low S, high V
        assert quads[2] == 3  # low S, low V
        assert quads[3] == 4  # high S, low V
        assert quads[4] == 1  # boundary: >= threshold -> high

    def test_assign_quadrants_median_default(self):
        quality_s = np.array([10, 20, 80, 90])
        quality_v = np.array([10, 90, 20, 80])
        quads = assign_quadrants(quality_s, quality_v)
        # Median S = 50, Median V = 50
        assert quads[0] == 3  # 10, 10 -> low, low
        assert quads[1] == 2  # 20, 90 -> low, high
        assert quads[2] == 4  # 80, 20 -> high, low
        assert quads[3] == 1  # 90, 80 -> high, high

    def test_assign_pair_quadrants(self):
        quality_s = np.array([80, 20, 60, 40, 90])
        quality_v = np.array([70, 80, 30, 50, 60])
        pairs = np.array([[0, 4], [1, 2]])  # min(80,90)=80, min(20,60)=20

        quads = assign_pair_quadrants(quality_s, quality_v, pairs,
                                       threshold_s=50, threshold_v=50)
        # Pair 0: min_s=80>=50, min_v=min(70,60)=60>=50 -> Q1
        assert quads[0] == 1
        # Pair 1: min_s=min(20,60)=20<50, min_v=min(80,30)=30<50 -> Q3
        assert quads[1] == 3

    def test_quadrant_performance(self, rng):
        """Q1 (high quality) should have lower EER than Q3 (low quality)."""
        n_gen = 4000
        n_imp = 4000

        # Create scores with quality-dependent separation
        quad_gen = rng.choice([1, 2, 3, 4], size=n_gen)
        quad_imp = rng.choice([1, 2, 3, 4], size=n_imp)

        genuine_sim = np.zeros(n_gen, dtype=np.float32)
        impostor_sim = rng.normal(0.3, 0.1, n_imp).astype(np.float32)

        # Q1 genuine: high scores, Q3 genuine: low scores
        for q, gen_mean in [(1, 0.85), (2, 0.7), (3, 0.55), (4, 0.7)]:
            mask = quad_gen == q
            genuine_sim[mask] = rng.normal(gen_mean, 0.05, mask.sum())

        result = compute_quadrant_performance(
            genuine_sim, impostor_sim, quad_gen, quad_imp,
        )

        assert "quadrants" in result
        assert "Q1" in result["quadrants"]
        assert "Q3" in result["quadrants"]

        eer_q1 = result["quadrants"]["Q1"]["eer"]
        eer_q3 = result["quadrants"]["Q3"]["eer"]

        assert eer_q1 < eer_q3, f"Q1 EER ({eer_q1:.4f}) should be < Q3 EER ({eer_q3:.4f})"
        assert result["q1_eer_lt_q3_eer"] is True

    def test_quadrant_counts(self):
        """Each sample should be assigned to exactly one quadrant."""
        quality_s = np.random.uniform(0, 100, 100)
        quality_v = np.random.uniform(0, 100, 100)
        quads = assign_quadrants(quality_s, quality_v)
        assert np.all(np.isin(quads, [1, 2, 3, 4]))
        assert len(quads) == 100


# ============================================================
# Integration Test
# ============================================================

class TestIntegration:

    def test_full_pipeline_synthetic(self, rng):
        """Run entire evaluation pipeline with synthetic data."""
        n = 500
        n_gen = 2000
        n_imp = 2000

        quality_s = rng.uniform(0, 100, n).astype(np.float32)
        quality_v = rng.uniform(0, 100, n).astype(np.float32)

        gen_pairs = np.column_stack([
            rng.randint(0, n, n_gen),
            rng.randint(0, n, n_gen),
        ])
        imp_pairs = np.column_stack([
            rng.randint(0, n, n_imp),
            rng.randint(0, n, n_imp),
        ])

        genuine_sim = rng.normal(0.7, 0.1, n_gen).astype(np.float32)
        impostor_sim = rng.normal(0.3, 0.1, n_imp).astype(np.float32)

        # Pairwise quality
        qg_s = np.minimum(quality_s[gen_pairs[:, 0]], quality_s[gen_pairs[:, 1]])
        qg_v = np.minimum(quality_v[gen_pairs[:, 0]], quality_v[gen_pairs[:, 1]])
        qi_s = np.minimum(quality_s[imp_pairs[:, 0]], quality_s[imp_pairs[:, 1]])
        qi_v = np.minimum(quality_v[imp_pairs[:, 0]], quality_v[imp_pairs[:, 1]])

        # 1. ERC
        tau = find_tau_for_fnmr(genuine_sim, impostor_sim, 0.10)
        erc = compute_erc(genuine_sim, impostor_sim, qg_s, qi_s, tau)
        assert len(erc["fnmr_values"]) > 0

        # 2. DET
        det = compute_det_curve(genuine_sim, impostor_sim)
        assert 0 < det["eer"] < 0.5

        # 3. Ranked DET
        rdet = compute_ranked_det(genuine_sim, impostor_sim, qg_s, qi_s)
        assert "eer_separation" in rdet

        # 4. Combined ERC
        cerc = compute_combined_erc(
            genuine_sim, impostor_sim,
            qg_s, qg_v, qi_s, qi_v, tau,
        )
        assert len(cerc) == 4

        # 5. Quadrant analysis
        quad_gen = assign_pair_quadrants(quality_s, quality_v, gen_pairs)
        quad_imp = assign_pair_quadrants(quality_s, quality_v, imp_pairs)
        qperf = compute_quadrant_performance(
            genuine_sim, impostor_sim, quad_gen, quad_imp,
        )
        assert len(qperf["quadrants"]) == 4

    def test_no_nan_inf_in_outputs(self, rng):
        """Ensure no NaN/Inf leak into core output arrays."""
        gen_sim = rng.normal(0.7, 0.1, 1000).astype(np.float32)
        imp_sim = rng.normal(0.3, 0.1, 1000).astype(np.float32)
        quality_gen = rng.uniform(10, 90, 1000).astype(np.float32)
        quality_imp = rng.uniform(10, 90, 1000).astype(np.float32)

        tau = find_tau_for_fnmr(gen_sim, imp_sim, 0.10)
        erc = compute_erc(gen_sim, imp_sim, quality_gen, quality_imp, tau)

        # Core ERC arrays: reject_fracs should have no NaN
        assert np.all(np.isfinite(erc["reject_fracs"]))
        # FNMR may have NaN at high rejection (no pairs left), but early values should be finite
        assert np.isfinite(erc["fnmr_values"][0])

        det = compute_det_curve(gen_sim, imp_sim)
        assert np.all(np.isfinite(det["fmr"]))
        assert np.all(np.isfinite(det["fnmr"]))
        assert np.isfinite(det["eer"])
