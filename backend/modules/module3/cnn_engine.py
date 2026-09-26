"""Module 3: Engine B — Raw-IQ 1D CNN AMC Engine.

Implements the validated lightweight deep learning architecture:
- Input shape: (128, 2) [Channel 0 = I, Channel 1 = Q]
- Architecture:
    Conv1D(64 filters, kernel=7, ReLU)
    BatchNorm1D
    MaxPool1D(2)
    Conv1D(128 filters, kernel=5, ReLU)
    BatchNorm1D
    MaxPool1D(2)
    Conv1D(128 filters, kernel=3, ReLU)
    BatchNorm1D
    GlobalAveragePooling1D
    Dense(64, ReLU)
    Dropout(0.3)
    Dense(7, Softmax)
- Parameter count: exactly 101,319 parameters
    (100,679 trainable weights + 640 BatchNorm moving statistics)

CRITICAL MODEL WEIGHTS RULE:
- Does not pretend a newly initialized network is the trained benchmark model.
- If pre-trained weights (cnn_model.pt) are absent, cleanly reports
  status = "MODEL_WEIGHTS_UNAVAILABLE".
"""

from typing import Any, Dict, List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

from .feature_extractor import MODULATION_CLASSES


class RawIQCNN(nn.Module):
    """PyTorch implementation of the validated 101,319-parameter AMC 1D CNN."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Block 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Block 3
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        # Classification Head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Accepts input of shape (batch, 128, 2) or (batch, 2, 128).
        """
        # If input is (batch, 128, 2), transpose to (batch, 2, 128) for PyTorch Conv1d
        if x.dim() == 3 and x.shape[1] == 128 and x.shape[2] == 2:
            x = x.permute(0, 2, 1)

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.softmax(self.fc2(x))
        return x

    def get_parameter_summary(self) -> Dict[str, int]:
        """Verify the exact parameter count against the research report."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        bn_stats = sum(b.numel() for name, b in self.named_buffers() if 'running' in name)
        total = trainable + bn_stats
        return {
            "trainable_parameters": trainable,
            "bn_running_parameters": bn_stats,
            "total_parameters": total
        }


class CNNEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), "artifacts", "cnn_model.pt"
        )
        self.classes = MODULATION_CLASSES
        self.device = torch.device("cpu")
        self.model = RawIQCNN(num_classes=len(self.classes))
        self.model.eval()
        self.is_weights_loaded = False
        self._try_load_weights()

    def _try_load_weights(self) -> bool:
        """Attempt to load trained weights from disk."""
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
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
        cnn_input: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Perform inference on the RMS-normalized array of shape (128, 2)."""
        if not self.is_available:
            return {
                "engine": "ENGINE_B_RAW_IQ_CNN",
                "status": "MODEL_WEIGHTS_UNAVAILABLE",
                "predicted_class": None,
                "probabilities": None,
                "parameter_count": 101319,
                "message": (
                    "Trained CNN weights (cnn_model.pt) not found. "
                    "Pre-trained weights must be placed in artifacts/ or trained on dataset."
                )
            }

        if cnn_input is None or cnn_input.shape != (128, 2):
            return {
                "engine": "ENGINE_B_RAW_IQ_CNN",
                "status": "INVALID_INPUT_SHAPE",
                "predicted_class": None,
                "probabilities": None,
                "parameter_count": 101319,
                "message": f"Expected input shape (128, 2), received {getattr(cnn_input, 'shape', None)}"
            }

        # Format into batch tensor (1, 128, 2)
        tensor_in = torch.from_numpy(cnn_input).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor_in)[0].cpu().numpy()

        probabilities = {
            cls_name: float(output[idx])
            for idx, cls_name in enumerate(self.classes)
        }
        best_class = max(probabilities, key=probabilities.get)

        return {
            "engine": "ENGINE_B_RAW_IQ_CNN",
            "status": "PREDICTION_SUCCESSFUL",
            "predicted_class": best_class,
            "probabilities": probabilities,
            "parameter_count": 101319
        }
