"""
Radio Signal Classifier for astroSETI

CNN + Transformer hybrid for classifying radio SETI spectrograms.
Uses 1D CNN backbone for spectral features and a small transformer
encoder for temporal pattern recognition.

Architecture:
    Input: 2D spectrogram (frequency x time)
    -> Conv1d layers along frequency axis (spectral features)
    -> Small Transformer encoder (2 layers, 4 heads) for temporal patterns
    -> MLP classification head -> 9 signal classes

References:
    - "Breakthrough Listen ML" (Zhang et al., 2018)
    - "Radio SETI Signal Classification" (Harp et al., 2019)
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SignalType(enum.Enum):
    """Types of radio signals encountered in SETI observations."""
    NARROWBAND_DRIFTING = 0
    NARROWBAND_STATIONARY = 1
    BROADBAND = 2
    PULSED = 3
    CHIRP = 4
    RFI_TERRESTRIAL = 5
    RFI_SATELLITE = 6
    NOISE = 7
    CANDIDATE_ET = 8


# Signal type labels for display
SIGNAL_LABELS: List[str] = [st.name.lower() for st in SignalType]

# RFI classes (Radio Frequency Interference)
RFI_CLASSES = {SignalType.RFI_TERRESTRIAL, SignalType.RFI_SATELLITE}

# Scientifically interesting classes
CANDIDATE_CLASSES = {SignalType.CANDIDATE_ET, SignalType.NARROWBAND_DRIFTING}


@dataclass
class ClassificationResult:
    """Output of signal classification."""
    signal_type: SignalType
    confidence: float
    rfi_probability: float
    feature_vector: np.ndarray
    all_scores: Dict[str, float] = field(default_factory=dict)


def _build_model(num_classes: int = 9, freq_bins: int = 256, time_steps: int = 64):
    """
    Build the CNN + Transformer hybrid model.

    Architecture:
        1. Conv1d backbone: spectral feature extraction along frequency axis
        2. Transformer encoder: temporal pattern detection across time steps
        3. MLP head: classification into signal types

    Args:
        num_classes: Number of output signal classes.
        freq_bins: Number of frequency bins in input spectrogram.
        time_steps: Number of time steps in input spectrogram.

    Returns:
        nn.Module: The assembled classification model.
    """
    import torch
    import torch.nn as nn

    class SpectralCNNBackbone(nn.Module):
        """1D CNN along the frequency axis to extract spectral features."""

        def __init__(self, freq_bins: int, embed_dim: int = 128):
            super().__init__()
            self.conv_layers = nn.Sequential(
                # Input: (batch, 1, freq_bins) per time step
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(64),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(16),
            )
            self.proj = nn.Linear(128 * 16, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, time_steps, freq_bins)
            Returns:
                (batch, time_steps, embed_dim)
            """
            batch, T, F = x.shape
            # Process each time step through the CNN
            x = x.reshape(batch * T, 1, F)  # (batch*T, 1, freq_bins)
            x = self.conv_layers(x)          # (batch*T, 128, 16)
            x = x.flatten(1)                 # (batch*T, 128*16)
            x = self.proj(x)                 # (batch*T, embed_dim)
            x = x.reshape(batch, T, -1)      # (batch, T, embed_dim)
            return x

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for temporal dimension."""

        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1), :]

    class SignalClassifierModel(nn.Module):
        """
        CNN + Transformer hybrid for radio signal classification.

        Pipeline:
            spectrogram -> SpectralCNN -> PositionalEncoding
            -> TransformerEncoder (2 layers, 4 heads)
            -> mean pooling -> MLP head -> class logits
        """

        def __init__(
            self,
            num_classes: int,
            freq_bins: int,
            time_steps: int,
            embed_dim: int = 128,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.freq_bins = freq_bins
            self.time_steps = time_steps
            self.embed_dim = embed_dim

            # Spectral CNN backbone
            self.cnn_backbone = SpectralCNNBackbone(freq_bins, embed_dim)

            # Positional encoding for temporal dimension
            self.pos_encoder = PositionalEncoding(embed_dim, max_len=time_steps)
            self.dropout = nn.Dropout(dropout)

            # Transformer encoder for temporal patterns
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers
            )

            # Classification head
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )

            # Feature projection (for feature_vector output)
            self.feature_proj = nn.Linear(embed_dim, embed_dim)

        def forward(
            self, x: torch.Tensor, return_features: bool = False
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Args:
                x: Input spectrogram (batch, freq_bins, time_steps).
                return_features: If True, also return the pooled feature vector.

            Returns:
                logits: (batch, num_classes)
                features: (batch, embed_dim) — only if return_features is True.
            """
            # Transpose to (batch, time_steps, freq_bins) for per-step CNN
            x = x.transpose(1, 2)

            # Spectral CNN: extract features per time step
            x = self.cnn_backbone(x)  # (batch, T, embed_dim)

            # Add positional encoding and apply transformer
            x = self.pos_encoder(x)
            x = self.dropout(x)
            x = self.transformer(x)  # (batch, T, embed_dim)

            # Mean pooling over time dimension
            features = x.mean(dim=1)  # (batch, embed_dim)

            # Classification
            logits = self.classifier(features)

            if return_features:
                proj_features = self.feature_proj(features)
                return logits, proj_features
            return logits

    return SignalClassifierModel(
        num_classes=num_classes,
        freq_bins=freq_bins,
        time_steps=time_steps,
    )


class SignalClassifier:
    """
    Radio signal classifier for SETI spectrograms.

    Uses a CNN + Transformer hybrid architecture:
    - 1D CNN backbone extracts spectral features along the frequency axis
    - Transformer encoder captures temporal patterns across time steps
    - MLP head produces class probabilities for 9 signal types

    Outputs:
    - Signal type classification (9 classes)
    - RFI probability estimate
    - 128-dim feature vector for downstream use
    - Per-class score dictionary
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        num_classes: int = len(SignalType),
        freq_bins: int = 256,
        time_steps: int = 64,
    ):
        """
        Initialize the signal classifier.

        Args:
            model_path: Path to pre-trained model weights (.pt file). If None,
                        creates a fresh (untrained) model.
            device: Target device ('cuda', 'mps', 'cpu'). Auto-detected if None.
            num_classes: Number of output signal classes.
            freq_bins: Expected number of frequency bins in input spectrograms.
            time_steps: Expected number of time steps in input spectrograms.
        """
        self.num_classes = num_classes
        self.freq_bins = freq_bins
        self.time_steps = time_steps
        self.device = device or self._detect_best_device()
        self.model = None

        self._load_model(model_path)

    def _detect_best_device(self) -> str:
        """
        Detect the best available device for inference.

        Priority: CUDA > MPS (Apple Silicon) > CPU.
        """
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not installed, falling back to CPU")
            return "cpu"

        if torch.cuda.is_available():
            logger.info("Using CUDA GPU acceleration")
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using Apple Metal (MPS) GPU acceleration")
            return "mps"

        logger.info("Using CPU (no GPU acceleration available)")
        return "cpu"

    def _load_model(self, model_path: Optional[str]) -> None:
        """Load pre-trained weights or create a default model."""
        import torch

        self.model = _build_model(
            num_classes=self.num_classes,
            freq_bins=self.freq_bins,
            time_steps=self.time_steps,
        )

        if model_path and Path(model_path).exists():
            logger.info(f"Loading pre-trained model from {model_path}")
            try:
                state_dict = torch.load(
                    model_path, map_location=self.device, weights_only=True
                )
                self.model.load_state_dict(state_dict)
                logger.info("Model weights loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model weights: {e}")
                logger.warning("Continuing with untrained model")
        else:
            if model_path:
                logger.warning(
                    f"Model path not found: {model_path}. Using untrained model."
                )
            else:
                logger.info("No model path provided. Created untrained model.")

        # Move to device with fallback
        try:
            self.model.to(self.device)
        except Exception as e:
            logger.warning(
                f"Failed to move model to {self.device}: {e}, falling back to CPU"
            )
            self.device = "cpu"
            self.model.to(self.device)

        self.model.eval()
        logger.info(
            f"SignalClassifier ready on {self.device} "
            f"({self.num_classes} classes, {self.freq_bins}x{self.time_steps} input)"
        )

    def _preprocess(self, spectrogram: np.ndarray) -> "torch.Tensor":
        """
        Preprocess a spectrogram for model input.

        Handles shape validation, normalization, and conversion to tensor.

        Args:
            spectrogram: 2D array of shape (frequency, time).

        Returns:
            Tensor of shape (1, freq_bins, time_steps) on the target device.

        Raises:
            ValueError: If the input is not a 2D array.
        """
        import torch
        from scipy.ndimage import zoom

        if spectrogram.ndim != 2:
            raise ValueError(
                f"Expected 2D spectrogram (freq x time), got shape {spectrogram.shape}"
            )

        freq, time = spectrogram.shape

        # Resize if dimensions don't match expected input
        if freq != self.freq_bins or time != self.time_steps:
            zoom_factors = (self.freq_bins / freq, self.time_steps / time)
            spectrogram = zoom(spectrogram, zoom_factors, order=1)

        # Normalize to zero mean, unit variance
        spec = spectrogram.astype(np.float32)
        mean = spec.mean()
        std = spec.std()
        if std > 0:
            spec = (spec - mean) / std
        else:
            spec = spec - mean

        # Convert to tensor: (1, freq_bins, time_steps)
        tensor = torch.from_numpy(spec).unsqueeze(0).to(self.device)
        return tensor

    def classify(self, spectrogram: np.ndarray) -> ClassificationResult:
        """
        Classify a single radio spectrogram.

        Args:
            spectrogram: 2D numpy array of shape (frequency_bins, time_steps).
                         Values should be power/intensity (linear or dB scale).

        Returns:
            ClassificationResult with signal type, confidence, RFI probability,
            feature vector, and per-class scores.

        Raises:
            ValueError: If the spectrogram is not 2D.
            RuntimeError: If inference fails (returns NOISE with low confidence).
        """
        import torch
        import torch.nn.functional as F

        try:
            tensor = self._preprocess(spectrogram)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

        try:
            with torch.no_grad():
                logits, features = self.model(tensor, return_features=True)
                probs = F.softmax(logits[0], dim=-1)
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            # Return a safe fallback result
            return ClassificationResult(
                signal_type=SignalType.NOISE,
                confidence=0.0,
                rfi_probability=0.0,
                feature_vector=np.zeros(128, dtype=np.float32),
                all_scores={st.name.lower(): 0.0 for st in SignalType},
            )

        prob_np = probs.cpu().numpy()
        top_idx = int(prob_np.argmax())
        top_conf = float(prob_np[top_idx])

        # Compute RFI probability (sum of RFI class probabilities)
        rfi_prob = float(
            prob_np[SignalType.RFI_TERRESTRIAL.value]
            + prob_np[SignalType.RFI_SATELLITE.value]
        )

        # Build per-class score dictionary
        all_scores = {
            st.name.lower(): float(prob_np[st.value]) for st in SignalType
        }

        return ClassificationResult(
            signal_type=SignalType(top_idx),
            confidence=top_conf,
            rfi_probability=rfi_prob,
            feature_vector=features[0].cpu().numpy(),
            all_scores=all_scores,
        )

    def classify_batch(
        self, spectrograms: List[np.ndarray]
    ) -> List[ClassificationResult]:
        """
        Classify multiple spectrograms in a batch (more efficient).

        Args:
            spectrograms: List of 2D numpy arrays, each (frequency, time).

        Returns:
            List of ClassificationResult, one per input spectrogram.
        """
        import torch
        import torch.nn.functional as F

        if not spectrograms:
            return []

        # Preprocess all and stack into a single batch
        tensors = [self._preprocess(s) for s in spectrograms]
        batch = torch.cat(tensors, dim=0)  # (N, freq_bins, time_steps)

        with torch.no_grad():
            logits, features = self.model(batch, return_features=True)
            probs = F.softmax(logits, dim=-1)

        results: List[ClassificationResult] = []
        for i in range(len(spectrograms)):
            prob_np = probs[i].cpu().numpy()
            top_idx = int(prob_np.argmax())

            rfi_prob = float(
                prob_np[SignalType.RFI_TERRESTRIAL.value]
                + prob_np[SignalType.RFI_SATELLITE.value]
            )
            all_scores = {
                st.name.lower(): float(prob_np[st.value]) for st in SignalType
            }

            results.append(
                ClassificationResult(
                    signal_type=SignalType(top_idx),
                    confidence=float(prob_np[top_idx]),
                    rfi_probability=rfi_prob,
                    feature_vector=features[i].cpu().numpy(),
                    all_scores=all_scores,
                )
            )

        return results

    @staticmethod
    def is_rfi(result: ClassificationResult) -> bool:
        """
        Check if a classification result indicates RFI.

        Args:
            result: Output from classify() or classify_batch().

        Returns:
            True if the signal is classified as terrestrial or satellite RFI.
        """
        return result.signal_type in RFI_CLASSES

    @staticmethod
    def is_candidate(
        result: ClassificationResult, rfi_threshold: float = 0.3
    ) -> bool:
        """
        Check if a classification result is a scientifically interesting candidate.

        A signal is considered a candidate if:
        - It is classified as CANDIDATE_ET, or
        - It is NARROWBAND_DRIFTING with low RFI probability.

        Args:
            result: Output from classify() or classify_batch().
            rfi_threshold: Maximum allowable RFI probability for NARROWBAND_DRIFTING
                           to qualify as a candidate.

        Returns:
            True if the signal warrants further investigation.
        """
        if result.signal_type == SignalType.CANDIDATE_ET:
            return True
        if (
            result.signal_type == SignalType.NARROWBAND_DRIFTING
            and result.rfi_probability < rfi_threshold
        ):
            return True
        return False
