"""Compute feature percentile lookup tables and category mapping for VQI GUI feedback.

Reads the 50K validation features, filters to selected features, and computes
percentile lookup tables (value at each percentile 0-100) for expert feedback.
Also builds a consolidated feature-to-category mapping.

Outputs:
  data/step9/feature_percentiles_s.npz  (430, 101)
  data/step9/feature_percentiles_v.npz  (133, 101)
  data/step9/feature_categories.json
"""

import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def load_selected_features(path):
    """Load selected feature names from text file."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def compute_percentile_table(features, feature_names, selected_names):
    """Compute percentile lookup table for selected features.

    Args:
        features: (N, D) array of all features
        feature_names: list of D feature names (all features)
        selected_names: list of selected feature names

    Returns:
        percentiles: (N_selected, 101) array
        selected_names: list of selected feature names (validated)
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    indices = []
    valid_names = []
    for name in selected_names:
        if name in name_to_idx:
            indices.append(name_to_idx[name])
            valid_names.append(name)
        else:
            print(f"  WARNING: Selected feature '{name}' not found in feature names")

    selected_data = features[:, indices]  # (N, N_selected)
    percentiles = np.percentile(selected_data, np.arange(101), axis=0).T  # (N_selected, 101)
    return percentiles, valid_names


def categorize_feature_s(name):
    """Map a VQI-S feature name to a consolidated category (7 categories)."""
    n = name.lower()
    # Noise-related
    if any(x in n for x in ["snr", "noise", "click", "dropout", "saturation",
                             "agc", "dnsmos", "nisqa"]):
        return "noise"
    # Reverberation
    if any(x in n for x in ["reverb", "rt60", "c50", "srmr", "drr"]):
        return "reverberation"
    # Spectral quality
    if any(x in n for x in ["spectral", "ltas", "alpha", "hammarberg",
                             "mfcc", "delta", "ac_", "autocorr",
                             "cepstral", "cpp"]):
        return "spectral"
    if any(x in n for x in ["sf_", "ss_", "sc_", "scf_", "shr_", "sflux_",
                             "se_", "sbw_", "skurt_", "sskew_"]):
        return "spectral"
    if n.startswith("frame") and any(x in n for x in ["sf", "ss", "sc", "sbw",
                                                       "sflux", "se", "sr_",
                                                       "skurt", "sskew", "scf",
                                                       "shr"]):
        return "spectral"
    # Dynamics
    if any(x in n for x in ["energy", "power", "onset", "clipping",
                             "dynamicrange", "peak"]):
        return "dynamics"
    if n.startswith("framee_") or n.startswith("framepc_"):
        return "dynamics"
    # Voice quality
    if any(x in n for x in ["hnr", "nhr", "f0", "pitch", "jitter", "shimmer",
                             "voice", "unvoiced", "tremor", "glottal",
                             "naq", "qoq", "hrf", "psp", "gci", "goi",
                             "gne", "h1h2", "h1a3", "mdvp", "avqi", "dsi",
                             "csid", "formant", "f1_", "f2_", "f3_",
                             "vocal"]):
        return "voice"
    # Temporal / prosodic
    if any(x in n for x in ["zcr", "speech", "pause", "rate", "turn",
                             "continuity", "interrupt", "duration",
                             "silence", "vad"]):
        return "temporal"
    # Channel artifacts
    if any(x in n for x in ["dc", "hum", "bandwidth", "quantiz", "musical",
                             "subband", "low_to_high", "lpc", "modulation"]):
        return "channel"
    return "other"


def categorize_feature_v(name):
    """Map a VQI-V feature name to a consolidated category (3 categories)."""
    n = name.lower()
    # Cepstral / spectral envelope
    if any(x in n for x in ["mfcc", "lpcc", "lfcc", "deltamfcc", "lsf",
                             "lpc", "reflect", "logarea", "ltfd",
                             "ltas", "mgdcc", "spectral"]):
        return "cepstral"
    # Formant / vocal tract
    if any(x in n for x in ["f1", "f2", "f3", "formant", "vocaltract"]):
        return "formant"
    # Prosodic / voice quality
    if any(x in n for x in ["f0", "speech", "rhythm", "pause", "jitter",
                             "shimmer", "hnr", "articulation", "tempo"]):
        return "prosodic"
    return "other"


def main():
    data_dir = os.path.join(BASE_DIR, "data")

    # Load feature names
    with open(os.path.join(data_dir, "step4", "features", "feature_names_s.json"), "r", encoding="utf-8") as f:
        feature_names_s = json.load(f)
    with open(os.path.join(data_dir, "step4", "features", "feature_names_v.json"), "r", encoding="utf-8") as f:
        feature_names_v = json.load(f)
    # V names may have empty first entry
    feature_names_v = [n for n in feature_names_v if n]

    print(f"All feature names: S={len(feature_names_s)}, V={len(feature_names_v)}")

    # Load selected features
    selected_s = load_selected_features(os.path.join(data_dir, "step5", "evaluation", "selected_features.txt"))
    selected_v = load_selected_features(os.path.join(data_dir, "step5", "evaluation_v", "selected_features.txt"))
    print(f"Selected features: S={len(selected_s)}, V={len(selected_v)}")

    # Load validation features
    print("Loading validation features...")
    features_s = np.load(os.path.join(data_dir, "step4", "features", "features_s_val.npy"))
    features_v = np.load(os.path.join(data_dir, "step4", "features", "features_v_val.npy"))
    print(f"Val features: S={features_s.shape}, V={features_v.shape}")

    # Compute percentile tables
    print("Computing VQI-S percentiles...")
    pct_s, names_s = compute_percentile_table(features_s, feature_names_s, selected_s)
    print(f"  VQI-S percentile table: {pct_s.shape}")

    print("Computing VQI-V percentiles...")
    pct_v, names_v = compute_percentile_table(features_v, feature_names_v, selected_v)
    print(f"  VQI-V percentile table: {pct_v.shape}")

    # Save percentile tables
    np.savez(
        os.path.join(data_dir, "step9", "feature_percentiles_s.npz"),
        percentiles=pct_s,
        feature_names=np.array(names_s),
    )
    np.savez(
        os.path.join(data_dir, "step9", "feature_percentiles_v.npz"),
        percentiles=pct_v,
        feature_names=np.array(names_v),
    )
    print("Saved percentile tables.")

    # Build category mapping
    categories = {}
    for name in names_s:
        categories[name] = categorize_feature_s(name)
    for name in names_v:
        categories[name] = categorize_feature_v(name)

    # Summary
    s_cats = {}
    for name in names_s:
        c = categories[name]
        s_cats[c] = s_cats.get(c, 0) + 1
    v_cats = {}
    for name in names_v:
        c = categories[name]
        v_cats[c] = v_cats.get(c, 0) + 1

    print(f"\nVQI-S category distribution: {s_cats}")
    print(f"VQI-V category distribution: {v_cats}")

    with open(os.path.join(data_dir, "step9", "feature_categories.json"), "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)
    print("Saved feature_categories.json")


if __name__ == "__main__":
    main()
