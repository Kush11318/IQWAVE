"""Module 3: Engine C — Multi-Scale Dilated CNN with Residual Connections.

Research-backed architecture from:
- Nature Scientific Reports 2026 (s41598-026-35558-7): Multi-scale kernels k=3,5,7
  with dilation rates d=1,2,4 for sequence classification
- IEEE Access 2021 Deep Architecture Survey: Residual connections for AMC
- UC San Diego 2025 (2503.04142v2): Deep ensemble for UQ + improved accuracy

Architecture:
    Input (128, 2)
    -> Multi-Scale Conv Block:
       Branch_k3: Conv1D(64, k=3, d=1) -> BN -> GELU
       Branch_k5: Conv1D(64, k=5, d=1) -> BN -> GELU
       Branch_k7: Conv1D(64, k=7, d=1) -> BN -> GELU
       -> Concat -> 192 channels
    -> Dilated Residual Block 1:
       d=1: Conv1D(128, k=3) -> BN -> GELU
       d=2: Conv1D(128, k=3) -> BN -> GELU
       d=4: Conv1D(128, k=3) -> BN -> GELU
       -> Concat + Residual projection -> 384 channels
    -> Global Average Pool -> 384-dim vector
    -> Dense(128, GELU) -> Dropout(0.3)
    -> Dense(7, Softmax)

For ensemble inference: trains 3 independently-seeded models and averages.

CRITICAL MODEL WEIGHTS RULE:
- If weights not found, cleanly reports status = "MODEL_WEIGHTS_UNAVAILABLE"
- Never fabricates predictions from an untrained network
"""

from typing import Any, Dict, List, Optional
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

from .feature_extractor import MODULATION_CLASSES


if TORCH_AVAILABLE:
    class MultiScaleDilatedCNN(nn.Module):
        """Research-backed multi-scale dilated CNN for AMC.

        Inspired by:
        - DBFCNN (Nature 2026): multi-scale kernels (k=3,5,7) + dilation rates
        - IEEE Access 2021 Survey: residual connections for AMC
        - GELU activation (shown to outperform ReLU for sequence tasks)
        """

        def __init__(self, num_classes: int = 7):
            super().__init__()

            # === Multi-Scale Stem Block ===
            # Three parallel branches with k=3, k=5, k=7
            # All with padding to preserve sequence length
            self.stem_k3 = nn.Sequential(
                nn.Conv1d(2, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.GELU()
            )
            self.stem_k5 = nn.Sequential(
                nn.Conv1d(2, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.GELU()
            )
            self.stem_k7 = nn.Sequential(
                nn.Conv1d(2, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.GELU()
            )
            # After concat: 192 channels
            self.stem_pool = nn.MaxPool1d(kernel_size=2)  # -> (batch, 192, 64)

            # === Dilated Residual Block ===
            # Three parallel dilated convolutions d=1,2,4
            self.dilated_d1 = nn.Sequential(
                nn.Conv1d(192, 96, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm1d(96),
                nn.GELU()
            )
            self.dilated_d2 = nn.Sequential(
                nn.Conv1d(192, 96, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm1d(96),
                nn.GELU()
            )
            self.dilated_d4 = nn.Sequential(
                nn.Conv1d(192, 96, kernel_size=3, padding=4, dilation=4),
                nn.BatchNorm1d(96),
                nn.GELU()
            )
            # After concat: 288 channels
            # Residual projection from 192 -> 288
            self.residual_proj = nn.Conv1d(192, 288, kernel_size=1)
            self.bn_res = nn.BatchNorm1d(288)

            # Second conv block (single scale, deeper)
            self.conv2 = nn.Sequential(
                nn.Conv1d(288, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2)   # -> (batch, 256, 32)
            )

            # === Global Average Pooling ===
            self.gap = nn.AdaptiveAvgPool1d(1)  # -> (batch, 256, 1)

            # === Classification Head ===
            self.head = nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
            # NOTE: No softmax here - use CrossEntropyLoss during training
            # Add softmax at inference time only
            self.num_classes = num_classes

        def extract_gap_features(self, x: Any) -> Any:
            """Extract 256-dimensional Global Average Pooling feature representation.

            Used for Phase 5 dual-branch late fusion head.
            """
            # Transpose to (batch, 2, 128) for Conv1d if needed
            if x.dim() == 3 and x.shape[1] == 128 and x.shape[2] == 2:
                x = x.permute(0, 2, 1)

            # Multi-scale stem
            s3 = self.stem_k3(x)   # (batch, 64, 128)
            s5 = self.stem_k5(x)   # (batch, 64, 128)
            s7 = self.stem_k7(x)   # (batch, 64, 128)
            stem_out = torch.cat([s3, s5, s7], dim=1)  # (batch, 192, 128)
            stem_out = self.stem_pool(stem_out)        # (batch, 192, 64)

            # Dilated residual block
            d1 = self.dilated_d1(stem_out)  # (batch, 96, 64)
            d2 = self.dilated_d2(stem_out)  # (batch, 96, 64)
            d4 = self.dilated_d4(stem_out)  # (batch, 96, 64)
            dilated_out = torch.cat([d1, d2, d4], dim=1)  # (batch, 288, 64)

            # Residual connection
            res = self.residual_proj(stem_out)  # (batch, 288, 64)
            res = self.bn_res(res)
            dilated_out = F.gelu(dilated_out + res)  # (batch, 288, 64)

            # Second conv block
            out = self.conv2(dilated_out)  # (batch, 256, 32)

            # GAP (256-dim feature representation)
            gap_feat = self.gap(out).squeeze(-1)  # (batch, 256)
            return gap_feat

        def forward(self, x: Any) -> Any:
            """Forward pass returning logits."""
            gap_feat = self.extract_gap_features(x)
            logits = self.head(gap_feat)  # (batch, 7)
            return logits

        def predict_proba(self, x: Any) -> Any:
            """Get softmax probabilities."""
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


class MultiScaleCNNEngine:
    """Multi-scale dilated CNN engine — single model variant.

    Loads pre-trained weights from multiscale_cnn.pt.
    Reports MODEL_WEIGHTS_UNAVAILABLE if weights not found.
    """

    WEIGHTS_FILENAME = "multiscale_cnn.pt"

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), "artifacts", self.WEIGHTS_FILENAME
        )
        self.classes = MODULATION_CLASSES
        self.is_weights_loaded = False

        if TORCH_AVAILABLE:
            self.device = torch.device("cpu")
            self.model = MultiScaleDilatedCNN(num_classes=len(self.classes))
            self.model.eval()
            self._try_load_weights()
        else:
            self.device = None
            self.model = None

    def _try_load_weights(self) -> bool:
        if not TORCH_AVAILABLE:
            return False
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.is_weights_loaded = True
                return True
            except Exception:
                self.is_weights_loaded = False
        return False

    @property
    def is_available(self) -> bool:
        return self.is_weights_loaded

    def predict(self, cnn_input: Optional[np.ndarray]) -> Dict[str, Any]:
        """Perform inference on the RMS-normalized array of shape (128, 2)."""
        if not self.is_available:
            return {
                "engine": "ENGINE_C_MULTISCALE_CNN",
                "status": "MODEL_WEIGHTS_UNAVAILABLE",
                "predicted_class": None,
                "probabilities": None,
                "message": f"Trained weights ({self.WEIGHTS_FILENAME}) not found in artifacts/"
            }

        if cnn_input is None or cnn_input.shape != (128, 2):
            return {
                "engine": "ENGINE_C_MULTISCALE_CNN",
                "status": "INVALID_INPUT_SHAPE",
                "predicted_class": None,
                "probabilities": None,
                "message": f"Expected (128, 2), got {getattr(cnn_input, 'shape', None)}"
            }

        tensor_in = torch.from_numpy(cnn_input.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs_tensor = self.model.predict_proba(tensor_in)[0].cpu().numpy()

        probabilities = {cls: float(probs_tensor[i]) for i, cls in enumerate(self.classes)}
        best_class = max(probabilities, key=probabilities.get)

        return {
            "engine": "ENGINE_C_MULTISCALE_CNN",
            "status": "PREDICTION_SUCCESSFUL",
            "predicted_class": best_class,
            "probabilities": probabilities
        }

    def extract_gap_features(self, cnn_input: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Extract 256-dim feature representation from the model."""
        if not self.is_available or cnn_input is None or cnn_input.shape != (128, 2):
            return None
        tensor_in = torch.from_numpy(cnn_input.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.extract_gap_features(tensor_in)[0].cpu().numpy()
        return feat


class CNNEnsembleEngine:
    """Deep ensemble of 3 independently-trained MultiScaleDilatedCNN models.

    Research basis: 2503.04142v2 (UC San Diego 2025)
    "Equal-weighted ensembles outperform single models and BNNs on AMC"
    - Diversity from independent random initialization + training
    - Provides uncertainty quantification via inter-model variance
    - CI widths: wide on uncertain/wrong predictions, narrow on confident/correct ones

    Weight files: ensemble_cnn_0.pt, ensemble_cnn_1.pt, ensemble_cnn_2.pt
    """

    N_MODELS = 3
    WEIGHT_FILES = ["ensemble_cnn_0.pt", "ensemble_cnn_1.pt", "ensemble_cnn_2.pt"]

    def __init__(self, artifacts_dir: Optional[str] = None):
        self.artifacts_dir = artifacts_dir or os.path.join(
            os.path.dirname(__file__), "artifacts"
        )
        self.classes = MODULATION_CLASSES
        self.models: List[Any] = []
        self._loaded_count = 0

        if TORCH_AVAILABLE:
            self.device = torch.device("cpu")
            self._try_load_all()
        else:
            self.device = None

    def _try_load_all(self) -> int:
        """Load all available ensemble member weights."""
        self.models = []
        self._loaded_count = 0

        for wf in self.WEIGHT_FILES:
            path = os.path.join(self.artifacts_dir, wf)
            m = MultiScaleDilatedCNN(num_classes=len(self.classes))
            m.eval()
            if os.path.exists(path):
                try:
                    state = torch.load(path, map_location=self.device, weights_only=True)
                    m.load_state_dict(state)
                    self.models.append(m)
                    self._loaded_count += 1
                except Exception:
                    pass
        return self._loaded_count

    @property
    def is_available(self) -> bool:
        return self._loaded_count >= 1

    @property
    def n_models_loaded(self) -> int:
        return self._loaded_count

    def predict(self, cnn_input: Optional[np.ndarray]) -> Dict[str, Any]:
        """Run ensemble prediction with uncertainty quantification.

        Returns averaged probabilities + inter-model variance as UQ signal.
        High variance = uncertain. Low variance = confident.
        """
        if not self.is_available:
            return {
                "engine": "ENGINE_C_CNN_ENSEMBLE",
                "status": "MODEL_WEIGHTS_UNAVAILABLE",
                "predicted_class": None,
                "probabilities": None,
                "uncertainty": None,
                "n_models": self._loaded_count,
                "message": "No ensemble member weights found. Run training script first."
            }

        if cnn_input is None or cnn_input.shape != (128, 2):
            return {
                "engine": "ENGINE_C_CNN_ENSEMBLE",
                "status": "INVALID_INPUT_SHAPE",
                "predicted_class": None,
                "probabilities": None,
                "uncertainty": None,
                "n_models": self._loaded_count,
                "message": f"Expected (128, 2), got {getattr(cnn_input, 'shape', None)}"
            }

        tensor_in = torch.from_numpy(cnn_input.astype(np.float32)).unsqueeze(0).to(self.device)

        # Collect predictions from each model
        all_probs = []
        with torch.no_grad():
            for m in self.models:
                p = m.predict_proba(tensor_in)[0].cpu().numpy()
                all_probs.append(p)

        all_probs_arr = np.array(all_probs)  # (n_models, 7)

        # Equal-weight averaging (best per 2025 paper)
        mean_probs = all_probs_arr.mean(axis=0)  # (7,)
        var_probs = all_probs_arr.var(axis=0)    # (7,) inter-model variance

        # Prediction entropy (aleatoric uncertainty)
        eps = 1e-9
        entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))
        max_entropy = float(np.log(len(self.classes)))
        normalized_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))

        # Confidence level thresholds (calibrated from literature)
        if normalized_entropy < 0.15:
            confidence_level = "HIGH"
        elif normalized_entropy < 0.40:
            confidence_level = "MEDIUM"
        elif normalized_entropy < 0.65:
            confidence_level = "LOW"
        else:
            confidence_level = "UNCERTAIN"

        probabilities = {cls: float(mean_probs[i]) for i, cls in enumerate(self.classes)}
        per_class_variance = {cls: float(var_probs[i]) for i, cls in enumerate(self.classes)}
        best_class = max(probabilities, key=probabilities.get)

        return {
            "engine": "ENGINE_C_CNN_ENSEMBLE",
            "status": "PREDICTION_SUCCESSFUL",
            "predicted_class": best_class,
            "probabilities": probabilities,
            "uncertainty": {
                "normalized_entropy": normalized_entropy,
                "confidence_level": confidence_level,
                "inter_model_variance": per_class_variance,
                "n_models_contributing": len(all_probs)
            },
            "n_models": self._loaded_count
        }

    def extract_gap_features(self, cnn_input: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Extract averaged 256-dim feature representation from all loaded ensemble members."""
        if not self.is_available or cnn_input is None or cnn_input.shape != (128, 2):
            return None
        tensor_in = torch.from_numpy(cnn_input.astype(np.float32)).unsqueeze(0).to(self.device)
        feats = []
        with torch.no_grad():
            for m in self.models:
                f = m.extract_gap_features(tensor_in)[0].cpu().numpy()
                feats.append(f)
        return np.mean(feats, axis=0) if feats else None
