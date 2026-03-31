"""Quick benchmark of each provider's inference speed."""
import torch
import time
import torchaudio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from implementation.vqi.providers import get_provider

# Load a sample Common Voice MP3
clips_dir = os.path.join(os.path.dirname(__file__), "..", "..", "Datasets",
                         "Common Voice", "cv-corpus-24.0-2025-12-05-en",
                         "cv-corpus-24.0-2025-12-05", "en", "clips")
sample = os.path.join(clips_dir, "common_voice_en_27250016.mp3")
waveform, sr = torchaudio.load(sample)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
waveform = waveform[0]  # mono
print(f"Sample: {waveform.shape[0]/16000:.1f}s")

batch = [waveform.clone() for _ in range(32)]

for pname in ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2"]:
    p = get_provider(pname, device="cuda")
    p.load_model()
    # Warmup
    p.extract_embedding_batch(batch[:2], 16000)
    torch.cuda.synchronize()

    t0 = time.time()
    p.extract_embedding_batch(batch, 16000)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"{pname}: {t1-t0:.2f}s for batch of 32 ({(t1-t0)/32*1000:.0f}ms/file)")
    del p
    torch.cuda.empty_cache()
