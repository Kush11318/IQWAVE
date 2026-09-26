"""Module 3: Engine A — 24-Feature Random Forest AMC Engine.

Implements the validated handcrafted feature classifier:
- 24 engineered features
- Classifier: sklearn.ensemble.RandomForestClassifier
- Configuration:
    n_estimators = 300
    min_samples_leaf = 2
    class_weight = "balanced"
    random_state = 42
- Supported classes (7):
    BPSK, QPSK, 8PSK, QAM16, QAM64, GFSK, CPFSK

CRITICAL MODEL WEIGHTS RULE:
- Trained model weights are not hard-coded or fabricated.
- If a serialized model artifact is not found on disk, the engine cleanly reports
  status = "MODEL_WEIGHTS_UNAVAILABLE".
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .feature_extractor import FEATURE_NAMES_24, MODULATION_CLASSES, feature_dict_to_vector


class RFEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), "artifacts", "rf_model.joblib"
        )
        self.classes = MODULATION_CLASSES
        self.config = {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42
        }
        self.model: Optional[RandomForestClassifier] = None
        self._try_load_model()

    def _try_load_model(self) -> bool:
        """Attempt to load serialized model artifact from disk."""
        if os.path.exists(self.model_path):
            try:
                loaded = joblib.load(self.model_path)
                if isinstance(loaded, RandomForestClassifier):
                    self.model = loaded
                    return True
            except Exception:
                self.model = None
        return False

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def build_unfitted_model(self) -> RandomForestClassifier:
        """Construct the Random Forest model with the exact validated configuration."""
        return RandomForestClassifier(
            n_estimators=self.config["n_estimators"],
            min_samples_leaf=self.config["min_samples_leaf"],
            class_weight=self.config["class_weight"],
            random_state=self.config["random_state"]
        )

    def train_and_save(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        save_path: Optional[str] = None
    ) -> RandomForestClassifier:
        """Train the Random Forest on supplied data and save the artifact."""
        model = self.build_unfitted_model()
        model.fit(X_train, y_train)
        target = save_path or self.model_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        joblib.dump(model, target)
        self.model = model
        return model

    def predict(
        self,
        features: Dict[str, Optional[float]]
    ) -> Dict[str, Any]:
        """Perform inference on the 24-dimensional feature vector.

        Returns structured prediction dictionary or reports MODEL_WEIGHTS_UNAVAILABLE.
        """
        if not self.is_available:
            return {
                "engine": "ENGINE_A_RANDOM_FOREST",
                "status": "MODEL_WEIGHTS_UNAVAILABLE",
                "predicted_class": None,
                "probabilities": None,
                "feature_count": len(FEATURE_NAMES_24),
                "message": (
                    "Trained Random Forest artifact (rf_model.joblib) not found. "
                    "Pre-trained weights must be placed in artifacts/ or trained on dataset."
                )
            }

        vec = feature_dict_to_vector(features).reshape(1, -1)
        proba = self.model.predict_proba(vec)[0]
        model_classes = list(self.model.classes_)

        # Map probabilities to standardized 7 classes
        probabilities: Dict[str, float] = {}
        for c in self.classes:
            if c in model_classes:
                idx = model_classes.index(c)
                probabilities[c] = float(proba[idx])
            else:
                probabilities[c] = 0.0

        best_class = max(probabilities, key=probabilities.get)

        return {
            "engine": "ENGINE_A_RANDOM_FOREST",
            "status": "PREDICTION_SUCCESSFUL",
            "predicted_class": best_class,
            "probabilities": probabilities,
            "feature_count": len(FEATURE_NAMES_24)
        }
