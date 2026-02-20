"""V66-V98: LP-derived identity features for VQI-V.

33 features from linear prediction analysis:
  V66-V79:  LSF means (14 coefficients)
  V80-V87:  Reflection coefficients (8)
  V88-V95:  Log Area Ratios (8)
  V96-V98:  LPC gain stats (mean, std, range)
"""

import numpy as np


def compute_lp_derived_features(waveform, sr, vad_mask, intermediates):
    features = {}
    lpc_list = intermediates["lpc_per_frame"]
    frames = intermediates["frames"]
    n_frames = frames.shape[1]
    vad = _align_mask(vad_mask, n_frames)

    lsf_all = []
    rc_all = []
    lar_all = []
    gain_all = []

    for i in range(n_frames):
        if not vad[i]:
            continue
        lpc_coeffs = lpc_list[i]
        if np.all(lpc_coeffs == 0):
            continue

        # LSF (Line Spectral Frequencies)
        lsf = _lpc_to_lsf(lpc_coeffs)
        if lsf is not None and len(lsf) >= 14:
            lsf_all.append(lsf[:14])

        # Reflection coefficients
        rc = _lpc_to_rc(lpc_coeffs)
        if rc is not None and len(rc) >= 8:
            rc_all.append(rc[:8])

        # Log Area Ratios
        if rc is not None and len(rc) >= 8:
            eps = 1e-12
            lar = np.log((1 + rc[:8]) / (1 - rc[:8] + eps) + eps)
            lar_all.append(lar)

        # LPC gain
        frame = frames[:, i]
        energy = np.sum(frame ** 2) / len(frame)
        from scipy.signal import lfilter
        residual = lfilter(lpc_coeffs, [1.0], frame)
        res_energy = np.sum(residual ** 2) / len(residual)
        gain_all.append(res_energy / (energy + 1e-12))

    # V66-V79: LSF means
    if lsf_all:
        lsf_arr = np.array(lsf_all)
        for i in range(14):
            features[f"V_LSF_Mean_{i+1}"] = float(np.mean(lsf_arr[:, i]))
    else:
        for i in range(14):
            features[f"V_LSF_Mean_{i+1}"] = 0.0

    # V80-V87: Reflection coefficients
    if rc_all:
        rc_arr = np.array(rc_all)
        for i in range(8):
            features[f"V_RC_{i+1}"] = float(np.mean(rc_arr[:, i]))
    else:
        for i in range(8):
            features[f"V_RC_{i+1}"] = 0.0

    # V88-V95: Log Area Ratios
    if lar_all:
        lar_arr = np.array(lar_all)
        for i in range(8):
            features[f"V_LAR_{i+1}"] = float(np.mean(lar_arr[:, i]))
    else:
        for i in range(8):
            features[f"V_LAR_{i+1}"] = 0.0

    # V96-V98: LPC gain stats
    if gain_all:
        gain_arr = np.array(gain_all)
        features["V_LPCGain_Mean"] = float(np.mean(gain_arr))
        features["V_LPCGain_Std"] = float(np.std(gain_arr))
        features["V_LPCGain_Range"] = float(np.max(gain_arr) - np.min(gain_arr))
    else:
        features["V_LPCGain_Mean"] = 0.0
        features["V_LPCGain_Std"] = 0.0
        features["V_LPCGain_Range"] = 0.0

    return features


def _lpc_to_lsf(lpc_coeffs):
    """Convert LPC to Line Spectral Frequencies."""
    p = len(lpc_coeffs) - 1
    if p < 2:
        return None

    a = lpc_coeffs
    # Form symmetric and antisymmetric polynomials
    p_poly = np.concatenate([a, [0]])
    q_poly = np.concatenate([[0], a[::-1]])
    p_sum = p_poly + q_poly
    p_diff = p_poly - q_poly

    # Find roots
    try:
        roots_sum = np.roots(p_sum)
        roots_diff = np.roots(p_diff)
    except Exception:
        return None

    # Extract angles on unit circle
    angles = []
    for r in np.concatenate([roots_sum, roots_diff]):
        if np.abs(np.abs(r) - 1.0) < 0.1:  # near unit circle
            angle = np.angle(r)
            if angle > 0:
                angles.append(angle)

    angles.sort()
    return np.array(angles) if len(angles) >= 14 else None


def _lpc_to_rc(lpc_coeffs):
    """Convert LPC to reflection coefficients using Levinson-Durbin recursion."""
    p = len(lpc_coeffs) - 1
    if p < 1:
        return None

    a = lpc_coeffs.copy()
    rc = np.zeros(p)

    for m in range(p, 0, -1):
        rc[m - 1] = a[m]
        if abs(rc[m - 1]) >= 1.0:
            rc[m - 1] = np.clip(rc[m - 1], -0.99, 0.99)
        denom = 1.0 - rc[m - 1] ** 2
        if abs(denom) < 1e-12:
            break
        a_prev = np.zeros(m)
        for i in range(1, m):
            a_prev[i] = (a[i] - rc[m - 1] * a[m - i]) / denom
        a[1:m] = a_prev[1:m]

    return rc


def _align_mask(vad_mask, n_frames):
    if len(vad_mask) == n_frames:
        return vad_mask.astype(bool)
    indices = np.round(np.linspace(0, len(vad_mask) - 1, n_frames)).astype(int)
    return vad_mask[indices].astype(bool)
