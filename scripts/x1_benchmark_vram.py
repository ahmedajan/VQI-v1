"""Benchmark VRAM usage per provider during inference."""
import torch
import torchaudio
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from implementation.vqi.providers import get_provider

clips_dir = os.path.join(os.path.dirname(__file__), "..", "..", "Datasets",
                         "Common Voice", "cv-corpus-24.0-2025-12-05-en",
                         "cv-corpus-24.0-2025-12-05", "en", "clips")
sample = os.path.join(clips_dir, "common_voice_en_27250016.mp3")
waveform, sr = torchaudio.load(sample)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
waveform = waveform[0]
print(f"Sample: {waveform.shape[0]/16000:.1f}s, {waveform.shape[0]} samples")

providers = {}
for pname in ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2"]:
    p = get_provider(pname, device="cuda")
    p.load_model()
    providers[pname] = p

print(f"\nModels loaded VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB")

for bs in [8, 16, 32]:
    print(f"\n--- Batch size {bs} ---")
    batch = [waveform.clone() for _ in range(bs)]
    for pname, p in providers.items():
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        p.extract_embedding_batch(batch, 16000)
        torch.cuda.synchronize()
        t1 = time.time()
        peak = torch.cuda.max_memory_allocated() / 1e6
        rate = bs / (t1 - t0)
        print(f"  {pname}: {t1-t0:.2f}s, peak VRAM: {peak:.0f} MB, {rate:.1f} files/s")
