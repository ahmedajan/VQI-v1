"""Custom exception classes for VQI preprocessing pipeline."""


class VQIError(Exception):
    """Base exception for all VQI errors."""
    pass


class AudioLoadError(VQIError):
    """Raised when audio file cannot be loaded (corrupt, unsupported format, empty)."""
    pass


class TooShortError(VQIError):
    """Raised when audio duration is below the minimum threshold (1.0s)."""
    pass


class InsufficientSpeechError(VQIError):
    """Raised when speech content after VAD is below minimum threshold."""
    pass
