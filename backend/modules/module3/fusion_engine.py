"""Module 3: Dual-Branch Feature Fusion Engine (Phase 5).

Based on DBFCNN: Dual-Branch Feature Fusion CNN (Nature Scientific Reports 2026):
- Branch 1: 32-dim Handcrafted physical/statistical features -> Dense(64, GELU)
- Branch 2: 256-dim Multi-scale CNN GAP sequence representations -> Dense(256, GELU)
- Fusion Layer: Concatenate (320-dim) -> Dense(128, GELU) -> Dropout(0.3) -> Dense(7)
- End-to-end trained fusion head replacing heuristic weighted averaging.

CRITICAL MODEL WEIGHTS RULE:
- Trained model weights are loaded from artifacts/fusion_head.pt.
- If weights not found, cleanly reports status = "MODEL_WEIGHTS_UNAVAILABLE".
- Never fabricates predictions from an untrained network.
"""

from typing import Any, Dict, List, Optional, Tuple
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

from .feature_extractor import FEATURE_NAMES_32, MODULATION_CLASSES, feature_dict_to_vector


if TORCH_AVAILABLE:
    class DualBranchFusionHead(nn.Module):
        """Dual-Branch late fusion network for AMC.

        Merges handcrafted expert domain features (32-dim) with deep multi-scale
        convolutional sequence representations (256-dim GAP).
        """

        def __init__(self, num_classes: int = 7, rf_dim: int = 32, cnn_dim: int = 256):
            super().__init__()
            self.rf_dim = rf_dim
            self.cnn_dim = cnn_dim
            self.num_classes = num_classes

            # Handcrafted feature branch
            self.rf_branch = nn.Sequential(
                nn.Linear(rf_dim, 64),
                nn.BatchNorm1d(64),
                nn.GELU()
            )

            # Deep representation branch
            self.cnn_branch = nn.Sequential(
                nn.Linear(cnn_dim, 256),
                nn.BatchNorm1d(256),
                nn.GELU()
            )

            # Fusion and classification head
            self.classifier = nn.Sequential(
                nn.Linear(64 + 256, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        def forward(self, rf_feats: Any, cnn_feats: Any) -> Any:
            """Forward pass returning raw class logits."""
            rf_out = self.rf_branch(rf_feats)
            cnn_out = self.cnn_branch(cnn_feats)
            fused = torch.cat([rf_out, cnn_out], dim=-1)
            logits = self.classifier(fused)
            return logits

        def predict_proba(self, rf_feats: Any, cnn_feats: Any) -> Any:
            """Softmax normalized probabilities."""
            logits = self.forward(rf_feats, cnn_feats)
            return F.softmax(logits, dim=-1)


class FusionEngine:
    """Trained Dual-Branch Fusion Engine for Module 3.

    Loads pre-trained weights from artifacts/fusion_head.pt.
    Reports MODEL_WEIGHTS_UNAVAILABLE if weights not found.
    """

    WEIGHTS_FILENAME = "fusion_head.pt"

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), "artifacts", self.WEIGHTS_FILENAME
        )
        self.classes = MODULATION_CLASSES
        self.is_weights_loaded = False

        if TORCH_AVAILABLE:
            self.device = torch.device("cpu")
            self.model = DualBranchFusionHead(num_classes=len(self.classes))
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

    def predict(
        self,
        rf_features: Any,
        cnn_gap_vector: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Perform dual-branch fused inference.

        Args:
            rf_features: Feature dict or 32-dim np.ndarray.
            cnn_gap_vector: 256-dim np.ndarray extracted from CNN ensemble or single CNN.

        Returns:
            Dictionary containing fused probabilities, calibrated confidence,
            entropy uncertainty, and prediction metadata.
        """
        if not self.is_available:
            return {
                "engine": "DUAL_BRANCH_FUSION",
                "status": "MODEL_WEIGHTS_UNAVAILABLE",
                "predicted_class": None,
                "probabilities": None,
                "confidence": None,
                "confidence_level": "UNCERTAIN",
                "message": f"Trained weights ({self.WEIGHTS_FILENAME}) not found in artifacts/"
            }

        # Format RF features to (1, 32)
        if isinstance(rf_features, dict):
            rf_vec = feature_dict_to_vector(rf_features, feature_names=FEATURE_NAMES_32)
        elif isinstance(rf_features, np.ndarray):
            rf_vec = rf_features.astype(np.float32)
        else:
            return {
                "engine": "DUAL_BRANCH_FUSION",
                "status": "INVALID_INPUT",
                "predicted_class": None,
                "probabilities": None,
                "confidence": None,
                "confidence_level": "UNCERTAIN",
                "message": "Invalid rf_features type"
            }

        if rf_vec.shape != (32,):
            return {
                "engine": "DUAL_BRANCH_FUSION",
                "status": "INVALID_INPUT_SHAPE",
                "predicted_class": None,
                "probabilities": None,
                "confidence": None,
                "confidence_level": "UNCERTAIN",
                "message": f"Expected 32 RF features, got {rf_vec.shape}"
            }

        if cnn_gap_vector is None or cnn_gap_vector.shape != (256,):
            return {
                "engine": "DUAL_BRANCH_FUSION",
                "status": "INVALID_INPUT_SHAPE",
                "predicted_class": None,
                "probabilities": None,
                "confidence": None,
                "confidence_level": "UNCERTAIN",
                "message": f"Expected 256 CNN GAP features, got {getattr(cnn_gap_vector, 'shape', None)}"
            }

        t_rf = torch.from_numpy(rf_vec).unsqueeze(0).to(self.device)
        t_cnn = torch.from_numpy(cnn_gap_vector.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(t_rf, t_cnn)[0].cpu().numpy()

        probabilities = {cls: float(probs[i]) for i, cls in enumerate(self.classes)}
        best_class = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[best_class])

        # Normalized entropy
        eps = 1e-9
        entropy = float(-np.sum(probs * np.log(probs + eps)))
        max_entropy = float(np.log(len(self.classes)))
        normalized_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))

        # Calibrated confidence levels per Research Plan Phase 6
        if normalized_entropy < 0.15:
            confidence_level = "HIGH"
        elif normalized_entropy < 0.40:
            confidence_level = "MEDIUM"
        elif normalized_entropy < 0.65:
            confidence_level = "LOW"
        else:
            confidence_level = "UNCERTAIN"

        return {
            "engine": "DUAL_BRANCH_FUSION",
            "status": "PREDICTION_SUCCESSFUL",
            "predicted_class": best_class,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "probabilities": probabilities,
            "uncertainty": {
                "normalized_entropy": normalized_entropy,
                "entropy": entropy
            }
        }
