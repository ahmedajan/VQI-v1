"""Speaker recognition providers for VQI scoring.

Train providers (P1-P3): ECAPA-TDNN, ResNet34, ECAPA2
Test providers  (P4-P5): x-vector, WavLM-SV
"""

from .base import SpeakerProvider
from .p1_ecapa import P1_ECAPA
from .p2_resnet import P2_RESNET
from .p3_ecapa2 import P3_ECAPA2
from .p4_xvector import P4_XVECTOR
from .p5_wavlm import P5_WAVLM

PROVIDERS = {
    "P1_ECAPA": P1_ECAPA,
    "P2_RESNET": P2_RESNET,
    "P3_ECAPA2": P3_ECAPA2,
    "P4_XVECTOR": P4_XVECTOR,
    "P5_WAVLM": P5_WAVLM,
}

TRAIN_PROVIDERS = ["P1_ECAPA", "P2_RESNET", "P3_ECAPA2"]
TEST_PROVIDERS = ["P4_XVECTOR", "P5_WAVLM"]


def get_provider(name: str, device: str = "cpu") -> SpeakerProvider:
    """Instantiate a provider by name.

    Args:
        name: Provider key, e.g. "P1_ECAPA".
        device: PyTorch device string ("cpu" or "cuda").

    Returns:
        An uninitialised SpeakerProvider instance. Call load_model() to use.
    """
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{name}'. Choose from: {list(PROVIDERS)}"
        )
    return PROVIDERS[name](device=device)
