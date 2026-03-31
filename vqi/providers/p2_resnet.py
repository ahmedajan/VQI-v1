"""P2: ResNet34 speaker embedding provider (256-dim).

Uses SpeechBrain's pretrained ResNet34 model trained on VoxCeleb.
Architecture: CNN + Squeeze-Excitation + Attentive Statistics Pooling.
"""

from typing import List

import numpy as np
import torch
import torchaudio

from .base import SpeakerProvider

SAVEDIR = "implementation/data/step1/pretrained/p2_resnet"


class P2_RESNET(SpeakerProvider):
    name = "P2_RESNET"
    embedding_dim = 256

    def load_model(self) -> None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-resnet-voxceleb",
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
        if waveform.dim() == 2:
            waveform = waveform[0:1, :]
        waveform = waveform.to(self.device)
        with torch.no_grad():
            emb = self._model.encode_batch(waveform)  # (1, 1, 256)
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
