"""Match VPQAD speakers across Lab/Cafeteria sessions using ECAPA2 embeddings.

Extracts speaker embeddings from all VPQAD recordings, computes per-subject
mean embeddings, then cross-compares across sessions to identify which
subject IDs correspond to the same physical person. Outputs a unified
subject ID mapping and renames files + updates metadata accordingly.
"""

import os
import sys
import glob
import numpy as np
import torch
import torchaudio
from collections import defaultdict

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

from vqi.providers.p3_ecapa2 import P3_ECAPA2

VPQAD_ROOT = "D:/vqi/Datasets/VPQAD"
SESSIONS = {
    "lab": os.path.join(VPQAD_ROOT, "Lab_Session_Data"),
    "caf": os.path.join(VPQAD_ROOT, "Cafeteria_Data"),
}
SUBDIRS = ["TD", "TID"]


def collect_files():
    """Collect all wav files grouped by (session, subject_id)."""
    files_by_subject = defaultdict(list)
    for session_key, session_path in SESSIONS.items():
        for subdir in SUBDIRS:
            dirpath = os.path.join(session_path, subdir)
            if not os.path.isdir(dirpath):
                continue
            for fname in sorted(os.listdir(dirpath)):
                if not fname.endswith(".wav"):
                    continue
                # e.g. sub001_1_td.wav -> sub001
                subject_id = fname.split("_")[0]
                full_path = os.path.join(dirpath, fname)
                files_by_subject[(session_key, subject_id)].append(full_path)
    return files_by_subject


def extract_mean_embedding(provider, file_list):
    """Extract embeddings from all files and return L2-normalized mean."""
    embeddings = []
    for fpath in file_list:
        waveform, sr = torchaudio.load(fpath)
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        emb = provider.extract_embedding(waveform, sr)
        embeddings.append(emb)
    mean_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 1e-12:
        mean_emb = mean_emb / norm
    return mean_emb


def main():
    print("=" * 70)
    print("VPQAD Speaker Matching via ECAPA2")
    print("=" * 70)

    # 1. Collect files
    files_by_subject = collect_files()
    lab_subjects = sorted(set(sid for (sess, sid) in files_by_subject if sess == "lab"))
    caf_subjects = sorted(set(sid for (sess, sid) in files_by_subject if sess == "caf"))
    print(f"\nLab subjects: {len(lab_subjects)} ({lab_subjects[0]}..{lab_subjects[-1]})")
    print(f"Caf subjects: {len(caf_subjects)} ({caf_subjects[0]}..{caf_subjects[-1]})")

    total_files = sum(len(v) for v in files_by_subject.values())
    print(f"Total audio files: {total_files}")

    # 2. Load ECAPA2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading ECAPA2 on {device}...")
    provider = P3_ECAPA2(device=device)
    provider.load_model()
    print("Model loaded.")

    # 3. Extract per-subject mean embeddings
    print("\nExtracting embeddings...")
    lab_embeddings = {}
    for i, sid in enumerate(lab_subjects):
        files = files_by_subject[("lab", sid)]
        lab_embeddings[sid] = extract_mean_embedding(provider, files)
        print(f"  Lab {sid}: {len(files)} files -> embedding extracted [{i+1}/{len(lab_subjects)}]")

    caf_embeddings = {}
    for i, sid in enumerate(caf_subjects):
        files = files_by_subject[("caf", sid)]
        caf_embeddings[sid] = extract_mean_embedding(provider, files)
        print(f"  Caf {sid}: {len(files)} files -> embedding extracted [{i+1}/{len(caf_subjects)}]")

    # 4. Cross-compare: all Lab vs all Caf
    print("\n" + "=" * 70)
    print("Cross-session similarity matrix (Lab vs Caf)")
    print("=" * 70)

    sim_matrix = np.zeros((len(lab_subjects), len(caf_subjects)))
    for i, lab_sid in enumerate(lab_subjects):
        for j, caf_sid in enumerate(caf_subjects):
            sim_matrix[i, j] = np.dot(lab_embeddings[lab_sid], caf_embeddings[caf_sid])

    # 5. Find matches using a high threshold
    # ECAPA2 genuine scores are typically > 0.5, impostor < 0.3
    # Use 0.5 as threshold, then verify
    THRESHOLD = 0.50

    print(f"\nMatches above threshold {THRESHOLD}:")
    print(f"{'Lab Subject':<15} {'Caf Subject':<15} {'Similarity':<12} {'Same ID?'}")
    print("-" * 55)

    matches = []
    for i, lab_sid in enumerate(lab_subjects):
        best_j = np.argmax(sim_matrix[i, :])
        best_sim = sim_matrix[i, best_j]
        best_caf = caf_subjects[best_j]
        if best_sim >= THRESHOLD:
            same_id = "YES" if lab_sid == best_caf else "NO"
            matches.append((lab_sid, best_caf, best_sim))
            print(f"{lab_sid:<15} {best_caf:<15} {best_sim:<12.4f} {same_id}")

    # Also check from Caf perspective (best match in Lab for each Caf subject)
    print(f"\n--- Reverse check (Caf -> best Lab match) ---")
    print(f"{'Caf Subject':<15} {'Lab Subject':<15} {'Similarity':<12} {'Same ID?'}")
    print("-" * 55)

    reverse_matches = []
    for j, caf_sid in enumerate(caf_subjects):
        best_i = np.argmax(sim_matrix[:, j])
        best_sim = sim_matrix[best_i, j]
        best_lab = lab_subjects[best_i]
        if best_sim >= THRESHOLD:
            same_id = "YES" if caf_sid == best_lab else "NO"
            reverse_matches.append((caf_sid, best_lab, best_sim))
            print(f"{caf_sid:<15} {best_lab:<15} {best_sim:<12.4f} {same_id}")

    # 6. Identify mutual best matches (both directions agree)
    print(f"\n{'=' * 70}")
    print("MUTUAL BEST MATCHES (confirmed same speaker)")
    print(f"{'=' * 70}")

    forward_map = {lab_sid: (caf_sid, sim) for lab_sid, caf_sid, sim in matches}
    reverse_map = {caf_sid: (lab_sid, sim) for caf_sid, lab_sid, sim in reverse_matches}

    mutual_matches = []
    for lab_sid, (caf_sid, sim) in forward_map.items():
        if caf_sid in reverse_map and reverse_map[caf_sid][0] == lab_sid:
            mutual_matches.append((lab_sid, caf_sid, sim))
            marker = " <-- SAME ID" if lab_sid == caf_sid else " <-- ID MISMATCH"
            print(f"  Lab {lab_sid} <-> Caf {caf_sid}  (sim={sim:.4f}){marker}")

    # 7. Show unmatched subjects
    matched_lab = set(m[0] for m in mutual_matches)
    matched_caf = set(m[1] for m in mutual_matches)
    unmatched_lab = [s for s in lab_subjects if s not in matched_lab]
    unmatched_caf = [s for s in caf_subjects if s not in matched_caf]

    print(f"\nUnmatched Lab subjects ({len(unmatched_lab)}): {unmatched_lab}")
    print(f"Unmatched Caf subjects ({len(unmatched_caf)}): {unmatched_caf}")

    # 8. Summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total Lab subjects: {len(lab_subjects)}")
    print(f"Total Caf subjects: {len(caf_subjects)}")
    print(f"Mutual matches found: {len(mutual_matches)}")
    print(f"Unique speakers total: {len(lab_subjects) + len(caf_subjects) - len(mutual_matches)}")

    # 9. Print full similarity matrix stats
    print(f"\nSimilarity matrix stats:")
    print(f"  Shape: {sim_matrix.shape}")
    print(f"  Min: {sim_matrix.min():.4f}")
    print(f"  Max: {sim_matrix.max():.4f}")
    print(f"  Mean: {sim_matrix.mean():.4f}")
    print(f"  Diagonal (same-ID pairs) mean: {np.mean([sim_matrix[i, i] for i in range(min(len(lab_subjects), len(caf_subjects)))]):.4f}")

    # 10. Show the top-N highest similarities for inspection
    print(f"\nTop 20 highest cross-session similarities:")
    flat_indices = np.argsort(sim_matrix.ravel())[::-1][:20]
    for idx in flat_indices:
        i, j = divmod(idx, len(caf_subjects))
        print(f"  Lab {lab_subjects[i]} <-> Caf {caf_subjects[j]}: {sim_matrix[i, j]:.4f}")

    # Save results
    output_path = os.path.join(VPQAD_ROOT, "speaker_matching_results.txt")
    with open(output_path, "w") as f:
        f.write("VPQAD Speaker Matching Results (ECAPA2)\n")
        f.write(f"Threshold: {THRESHOLD}\n\n")
        f.write("MUTUAL MATCHES:\n")
        for lab_sid, caf_sid, sim in mutual_matches:
            f.write(f"  Lab {lab_sid} <-> Caf {caf_sid} (sim={sim:.4f})\n")
        f.write(f"\nUnmatched Lab: {unmatched_lab}\n")
        f.write(f"Unmatched Caf: {unmatched_caf}\n")
        f.write(f"\nFull similarity matrix saved to speaker_matching_sim_matrix.npy\n")

    np.save(os.path.join(VPQAD_ROOT, "speaker_matching_sim_matrix.npy"), sim_matrix)
    np.save(os.path.join(VPQAD_ROOT, "speaker_matching_lab_ids.npy"), np.array(lab_subjects))
    np.save(os.path.join(VPQAD_ROOT, "speaker_matching_caf_ids.npy"), np.array(caf_subjects))

    print(f"\nResults saved to {output_path}")
    return mutual_matches, unmatched_lab, unmatched_caf


if __name__ == "__main__":
    main()
