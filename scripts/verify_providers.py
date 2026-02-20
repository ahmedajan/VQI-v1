"""Step 1.5 verification: load all 5 speaker recognition providers,
extract embeddings from test audio, and confirm correct behaviour.

Usage:
    python implementation/scripts/verify_providers.py
"""

import sys
import time
from pathlib import Path

import torch
import torchaudio

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "implementation"))

from vqi.providers import PROVIDERS, get_provider

# ---------------------------------------------------------------------------
# Test audio paths
# ---------------------------------------------------------------------------
SAME_SPK_1 = PROJECT_ROOT / "Datasets" / "voxCELEB1" / "wav" / "id10270" / "5r0dWxy17C8" / "00001.wav"
SAME_SPK_2 = PROJECT_ROOT / "Datasets" / "voxCELEB1" / "wav" / "id10270" / "5r0dWxy17C8" / "00002.wav"
DIFF_SPK_DIR = PROJECT_ROOT / "Datasets" / "voxCELEB1" / "wav" / "id10271"


def find_first_wav(directory: Path) -> Path:
    """Return the first .wav file found under *directory*."""
    for wav in sorted(directory.rglob("*.wav")):
        return wav
    raise FileNotFoundError(f"No WAV files found under {directory}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Load test audio
    wav1, sr1 = torchaudio.load(str(SAME_SPK_1))
    wav2, sr2 = torchaudio.load(str(SAME_SPK_2))
    diff_wav_path = find_first_wav(DIFF_SPK_DIR)
    wav3, sr3 = torchaudio.load(str(diff_wav_path))

    print(f"Same-speaker A : {SAME_SPK_1.name}  ({wav1.shape[1]/sr1:.1f}s)")
    print(f"Same-speaker B : {SAME_SPK_2.name}  ({wav2.shape[1]/sr2:.1f}s)")
    print(f"Diff-speaker   : {diff_wav_path.name}  ({wav3.shape[1]/sr3:.1f}s)")
    print()

    results = []
    all_ok = True

    for name in PROVIDERS:
        print(f"{'='*60}")
        print(f"Loading {name} ...")
        provider = get_provider(name, device=device)

        t0 = time.time()
        provider.load_model()
        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s")

        # Extract embeddings
        t0 = time.time()
        emb1 = provider.extract_embedding(wav1, sr1)
        emb2 = provider.extract_embedding(wav2, sr2)
        emb3 = provider.extract_embedding(wav3, sr3)
        embed_time = time.time() - t0

        # Checks
        dim_ok = emb1.shape == (provider.embedding_dim,)
        genuine = provider.compute_similarity(emb1, emb2)
        impostor = provider.compute_similarity(emb1, emb3)
        order_ok = genuine > impostor
        range_ok = -1.0 <= genuine <= 1.0 and -1.0 <= impostor <= 1.0

        status = "PASS" if (dim_ok and order_ok and range_ok) else "FAIL"
        if status == "FAIL":
            all_ok = False

        print(f"  Dim: {emb1.shape} (expected ({provider.embedding_dim},)) -> {'OK' if dim_ok else 'FAIL'}")
        print(f"  Genuine sim:  {genuine:+.4f}")
        print(f"  Impostor sim: {impostor:+.4f}")
        print(f"  Genuine > Impostor: {'OK' if order_ok else 'FAIL'}")
        print(f"  Range [-1,1]: {'OK' if range_ok else 'FAIL'}")
        print(f"  Embed time: {embed_time:.2f}s (3 utterances)")
        print(f"  Status: {status}")
        print()

        results.append({
            "name": name,
            "dim": emb1.shape[0],
            "genuine": genuine,
            "impostor": impostor,
            "status": status,
        })

    # Summary table
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Provider':<14} {'Dim':>5} {'Genuine':>9} {'Impostor':>9} {'Status':>8}")
    print(f"{'-'*14} {'-'*5} {'-'*9} {'-'*9} {'-'*8}")
    for r in results:
        print(
            f"{r['name']:<14} {r['dim']:>5} {r['genuine']:>+9.4f} "
            f"{r['impostor']:>+9.4f} {r['status']:>8}"
        )
    print()

    if all_ok:
        print("All 5 providers PASSED verification.")
    else:
        print("SOME PROVIDERS FAILED. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
