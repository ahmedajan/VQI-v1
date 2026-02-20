"""Step 8.4: Computation efficiency benchmarks.

Measures end-to-end VQI scoring latency for 3s, 10s, 60s audio.
Per-component breakdown: load, preprocess, VAD, frame features,
global features, VQI-V features, RF predict.

Targets (single-core CPU, no GPU):
  - 3s audio:  < 50ms
  - 10s audio: < 100ms
  - 60s audio: < 300ms

Usage:
    python scripts/benchmark_speed.py [--n-runs 10]
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPL_DIR = os.path.join(_SCRIPT_DIR, "..")
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(_IMPL_DIR, "data")
MODELS_DIR = os.path.join(_IMPL_DIR, "models")
EVAL_DIR = os.path.join(DATA_DIR, "evaluation")
EVAL_V_DIR = os.path.join(DATA_DIR, "evaluation_v")
OUTPUT_DIR = os.path.join(DATA_DIR, "test_scores")

DURATIONS = [3, 10, 60]  # seconds
SAMPLE_RATE = 16000
TARGETS_MS = {3: 50, 10: 100, 60: 300}


def generate_synthetic_audio(duration_s, sr=16000):
    """Generate synthetic speech-like audio for benchmarking."""
    n_samples = int(duration_s * sr)
    rng = np.random.RandomState(42)
    # White noise + some tonal content to simulate speech
    t = np.arange(n_samples) / sr
    signal = (
        0.3 * rng.randn(n_samples)
        + 0.5 * np.sin(2 * np.pi * 150 * t)  # F0-like
        + 0.2 * np.sin(2 * np.pi * 500 * t)  # formant-like
    )
    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal.astype(np.float32)


def benchmark_single(waveform, sr, feature_names_s, feature_names_v,
                     selected_s, selected_v, clf_s, clf_v, n_runs=10):
    """Benchmark a single audio duration.

    Returns dict with per-component and total timings.
    """
    from vqi.preprocessing.normalize import dc_remove_and_normalize
    from vqi.preprocessing.vad import energy_vad
    from vqi.core.feature_orchestrator import compute_all_features
    from vqi.core.feature_orchestrator_v import compute_all_features_v
    from vqi.prediction.random_forest import predict_score as predict_s
    from vqi.prediction.random_forest_v import predict_score as predict_v

    name_to_idx_s = {n: i for i, n in enumerate(feature_names_s)}
    name_to_idx_v = {n: i for i, n in enumerate(feature_names_v)}
    idx_s = [name_to_idx_s[n] for n in selected_s]
    idx_v = [name_to_idx_v[n] for n in selected_v]

    timings = {k: [] for k in [
        "preprocess", "vad", "frame_features", "global_features",
        "vqi_v_features", "rf_predict", "total",
    ]}

    for run_i in range(n_runs):
        t_total_start = time.perf_counter()

        # Preprocess
        t0 = time.perf_counter()
        wf = dc_remove_and_normalize(waveform)
        t_preprocess = time.perf_counter() - t0

        # VAD
        t0 = time.perf_counter()
        vad_mask, _, _ = energy_vad(wf)
        t_vad = time.perf_counter() - t0

        # Frame + global features (computed together in orchestrator)
        t0 = time.perf_counter()
        _, feat_arr_s, intermediates = compute_all_features(
            wf, sr, vad_mask, raw_waveform=waveform
        )
        t_features = time.perf_counter() - t0

        # VQI-V features
        t0 = time.perf_counter()
        _, feat_arr_v = compute_all_features_v(wf, sr, vad_mask, intermediates)
        t_v_features = time.perf_counter() - t0

        # RF predict
        t0 = time.perf_counter()
        x_s = feat_arr_s[idx_s]
        x_v = feat_arr_v[idx_v]
        score_s = predict_s(clf_s, x_s)
        score_v = predict_v(clf_v, x_v)
        t_predict = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        timings["preprocess"].append(t_preprocess * 1000)
        timings["vad"].append(t_vad * 1000)
        # Split frame vs global is not separable from orchestrator, report combined
        timings["frame_features"].append(t_features * 1000)
        timings["global_features"].append(0.0)  # included in frame_features
        timings["vqi_v_features"].append(t_v_features * 1000)
        timings["rf_predict"].append(t_predict * 1000)
        timings["total"].append(t_total * 1000)

    # Compute stats
    stats = {}
    for k, vals in timings.items():
        arr = np.array(vals)
        stats[k] = {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="VQI speed benchmarks")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of runs per duration")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load models and feature names
    import json as json_mod
    import joblib

    with open(os.path.join(DATA_DIR, "features", "feature_names_s.json"), "r", encoding="utf-8") as f:
        feature_names_s = json_mod.load(f)
    with open(os.path.join(DATA_DIR, "features", "feature_names_v.json"), "r", encoding="utf-8") as f:
        feature_names_v = json_mod.load(f)

    with open(os.path.join(EVAL_DIR, "selected_features.txt"), "r", encoding="utf-8") as f:
        selected_s = [line.strip() for line in f if line.strip()]
    with open(os.path.join(EVAL_V_DIR, "selected_features.txt"), "r", encoding="utf-8") as f:
        selected_v = [line.strip() for line in f if line.strip()]

    clf_s = joblib.load(os.path.join(MODELS_DIR, "vqi_rf_model.joblib"))
    clf_v = joblib.load(os.path.join(MODELS_DIR, "vqi_v_rf_model.joblib"))

    logger.info(f"Loaded models: S ({clf_s.n_estimators} trees, {len(selected_s)} features), "
                f"V ({clf_v.n_estimators} trees, {len(selected_v)} features)")

    # Warmup run
    logger.info("Warmup run...")
    warmup = generate_synthetic_audio(3, SAMPLE_RATE)
    benchmark_single(warmup, SAMPLE_RATE, feature_names_s, feature_names_v,
                     selected_s, selected_v, clf_s, clf_v, n_runs=2)

    # Benchmark each duration
    all_results = {}
    for dur in DURATIONS:
        logger.info(f"\nBenchmarking {dur}s audio ({args.n_runs} runs)...")
        waveform = generate_synthetic_audio(dur, SAMPLE_RATE)
        stats = benchmark_single(
            waveform, SAMPLE_RATE, feature_names_s, feature_names_v,
            selected_s, selected_v, clf_s, clf_v, n_runs=args.n_runs,
        )

        target = TARGETS_MS[dur]
        total_mean = stats["total"]["mean_ms"]
        passed = total_mean < target

        stats["duration_s"] = dur
        stats["target_ms"] = target
        stats["passed"] = passed

        all_results[f"{dur}s"] = stats

        logger.info(f"  Total: {total_mean:.1f} +/- {stats['total']['std_ms']:.1f} ms "
                    f"(target: <{target}ms) -> {'PASS' if passed else 'FAIL'}")
        logger.info(f"  Breakdown:")
        for component in ["preprocess", "vad", "frame_features", "vqi_v_features", "rf_predict"]:
            if stats[component]["mean_ms"] > 0:
                logger.info(f"    {component}: {stats[component]['mean_ms']:.1f} ms")

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved: {output_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    all_pass = True
    for dur in DURATIONS:
        key = f"{dur}s"
        r = all_results[key]
        status = "PASS" if r["passed"] else "FAIL"
        all_pass = all_pass and r["passed"]
        logger.info(f"  {dur:3d}s audio: {r['total']['mean_ms']:6.1f}ms (target: <{r['target_ms']}ms) [{status}]")

    logger.info(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAIL'}")


if __name__ == "__main__":
    main()
