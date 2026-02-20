"""Abstract base class for speaker recognition providers."""

from abc import ABC, abstractmethod

import numpy as np
import torch


class SpeakerProvider(ABC):
    """Base class for all speaker embedding providers.

    Each provider wraps a pretrained speaker recognition model,
    exposing a uniform interface for embedding extraction and
    similarity scoring.
    """

    name: str
    embedding_dim: int

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @abstractmethod
    def load_model(self) -> None:
        """Download (if needed) and load the pretrained model."""
        ...

    @abstractmethod
    def extract_embedding(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> np.ndarray:
        """Extract a speaker embedding from an audio waveform.

        Args:
            waveform: Audio tensor of shape (channels, samples).
            sample_rate: Sample rate of the waveform in Hz.

        Returns:
            L2-normalised numpy array of shape (embedding_dim,).
        """
        ...

    def extract_embedding_batch(
        self, waveforms: list[torch.Tensor], sample_rate: int
    ) -> np.ndarray:
        """Extract embeddings for a batch of waveforms.

        Default implementation loops over individual extraction.
        Override for true batched inference.

        Args:
            waveforms: List of 1D tensors (samples,), all at sample_rate.
            sample_rate: Common sample rate in Hz.

        Returns:
            L2-normalised numpy array of shape (batch, embedding_dim).
        """
        results = []
        for wv in waveforms:
            results.append(self.extract_embedding(wv.unsqueeze(0), sample_rate))
        return np.stack(results)

    @staticmethod
    def _l2_normalize_batch(x: np.ndarray) -> np.ndarray:
        """L2-normalize each row of a 2D array."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return x / norms

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised embeddings."""
        return float(np.dot(emb1, emb2))

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(
                f"{self.name}: model not loaded. Call load_model() first."
            )

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            return x
        return x / norm
