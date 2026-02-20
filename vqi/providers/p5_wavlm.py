"""P5: WavLM-SV speaker embedding provider (512-dim).

Uses Microsoft's WavLM-Base-Plus fine-tuned for speaker verification.
Architecture: SSL Transformer + x-vector head.
"""

import numpy as np
import torch
import torchaudio

from .base import SpeakerProvider

MODEL_ID = "microsoft/wavlm-base-plus-sv"
CACHE_DIR = "implementation/data/pretrained/p5_wavlm"


class P5_WAVLM(SpeakerProvider):
    name = "P5_WAVLM"
    embedding_dim = 512

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self._feature_extractor = None

    def load_model(self) -> None:
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            MODEL_ID, cache_dir=CACHE_DIR
        )
        self._model = WavLMForXVector.from_pretrained(
            MODEL_ID, cache_dir=CACHE_DIR
        )
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
        # Feature extractor expects 1-D numpy array
        if waveform.dim() == 2:
            waveform = waveform[0]  # (samples,)
        audio_np = waveform.cpu().numpy()

        inputs = self._feature_extractor(
            audio_np, sampling_rate=16000, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
        emb = outputs.embeddings.squeeze().cpu().numpy()  # (512,)
        return self._l2_normalize(emb)

    def extract_embedding_batch(
        self, waveforms: list[torch.Tensor], sample_rate: int
    ) -> np.ndarray:
        self._ensure_loaded()
        audio_list = []
        for wv in waveforms:
            if sample_rate != 16000:
                wv = torchaudio.functional.resample(wv, sample_rate, 16000)
            if wv.dim() == 2:
                wv = wv[0]
            audio_list.append(wv.cpu().numpy())
        inputs = self._feature_extractor(
            audio_list, sampling_rate=16000, return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        embs = outputs.embeddings.cpu().numpy()  # (batch, 512)
        return self._l2_normalize_batch(embs)
