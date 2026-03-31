"""P1: ECAPA-TDNN speaker embedding provider (192-dim).

Uses SpeechBrain's pretrained ECAPA-TDNN model trained on VoxCeleb.
Architecture: TDNN + Squeeze-Excitation + Attentive Statistics Pooling.
"""

from typing import List

import numpy as np
import torch
import torchaudio

from .base import SpeakerProvider

SAVEDIR = "implementation/data/step1/pretrained/p1_ecapa"


class P1_ECAPA(SpeakerProvider):
    name = "P1_ECAPA"
    embedding_dim = 192

    def load_model(self) -> None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=SAVEDIR,
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )

    def extract_embedding(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> np.ndarray:
        self._ensure_loaded()
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=16000
            )
        # SpeechBrain expects (batch, samples) — take first channel
        if waveform.dim() == 2:
            waveform = waveform[0:1, :]  # (1, samples)
        waveform = waveform.to(self.device)
        with torch.no_grad():
            emb = self._model.encode_batch(waveform)  # (1, 1, 192)
        emb = emb.squeeze().cpu().numpy()
        return self._l2_normalize(emb)

    def extract_embedding_batch(
        self, waveforms: List[torch.Tensor], sample_rate: int
    ) -> np.ndarray:
        self._ensure_loaded()
        processed = []
        for wv in waveforms:
            if sample_rate != 16000:
                wv = torchaudio.functional.resample(wv, sample_rate, 16000)
            if wv.dim() == 2:
                wv = wv[0]
            elif wv.dim() == 0:
                wv = wv.unsqueeze(0)
            processed.append(wv)
        lengths = [w.shape[0] for w in processed]
        max_len = max(lengths)
        batch = torch.zeros(len(processed), max_len)
        for i, w in enumerate(processed):
            batch[i, :w.shape[0]] = w
        wav_lens = torch.tensor([l / max_len for l in lengths], dtype=torch.float32)
        batch = batch.to(self.device)
        wav_lens = wav_lens.to(self.device)
        with torch.no_grad():
            embs = self._model.encode_batch(batch, wav_lens)
        embs = embs.squeeze(1).cpu().numpy()
        return self._l2_normalize_batch(embs)
