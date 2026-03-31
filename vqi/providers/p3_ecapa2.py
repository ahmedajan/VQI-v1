"""P3: ECAPA2 speaker embedding provider (192-dim).

Uses the Jenthe/ECAPA2 TorchScript model from HuggingFace.
Architecture: Hybrid 1D+2D Convolution + TDNN.
SOTA: 0.17% EER on VoxCeleb1-O.

Note: The TorchScript model must be loaded on CPU first (map_location="cpu"),
then moved to the target device. GPU JIT fusion must be disabled to avoid a
kernel compilation error with complex-valued STFT on some CUDA toolkits.
"""

import numpy as np
import torch
import torchaudio

from .base import SpeakerProvider

CACHE_DIR = "implementation/data/step1/pretrained/p3_ecapa2"


class P3_ECAPA2(SpeakerProvider):
    name = "P3_ECAPA2"
    embedding_dim = 192

    def load_model(self) -> None:
        from huggingface_hub import hf_hub_download

        # Disable GPU JIT fusion to avoid complex-valued STFT kernel error
        torch._C._jit_override_can_fuse_on_gpu(False)

        model_path = hf_hub_download(
            repo_id="Jenthe/ECAPA2",
            filename="ecapa2.pt",
            cache_dir=CACHE_DIR,
        )
        # Load on CPU first, then move to target device
        self._model = torch.jit.load(model_path, map_location="cpu")
        if self.device != "cpu":
            self._model = self._model.to(self.device)
        self._model.eval()

    def extract_embedding(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> np.ndarray:
        self._ensure_loaded()
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=16000
            )
        # ECAPA2 expects (batch, samples) — mono
        if waveform.dim() == 2:
            waveform = waveform[0:1, :]  # (1, samples)
        waveform = waveform.to(self.device)
        with torch.no_grad():
            emb = self._model(waveform)  # (1, 192)
        emb = emb.squeeze().cpu().numpy()
        return self._l2_normalize(emb)

    # NOTE: Batched extraction is intentionally NOT overridden here.
    # The TorchScript ECAPA2 model has severe performance degradation with
    # padded batches (6x slower than individual due to internal STFT).
    # The base class fallback (loop over individual) is faster.
