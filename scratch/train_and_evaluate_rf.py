import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from backend.modules.module3.feature_extractor import (
    FEATURE_NAMES_24,
    MODULATION_CLASSES,
    extract_24_features,
    feature_dict_to_vector
)
from backend.modules.module3.rf_engine import RFEngine


def generate_pulse(sps: int = 4, beta: float = 0.35, span: int = 8) -> np.ndarray:
    """Generate Root Raised Cosine (RRC) pulse."""
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    h = np.zeros(len(t), dtype=np.float64)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif beta > 0 and np.isclose(abs(ti), 1.0 / (4.0 * beta)):
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta)) +
                (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            denom = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            if abs(denom) > 1e-12:
                numer = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
                h[i] = numer / denom
            else:
                h[i] = 1.0
    h /= np.sqrt(np.sum(h ** 2))
    return h


def generate_signal(mod: str, n_samples: int = 512, sps: int = 4, snr_db: float = 20.0, cfo_norm: float = 0.01) -> np.ndarray:
    """Synthesize baseband I/Q signal for a specified modulation class."""
    rng = np.random.default_rng()
    n_symbols = int(np.ceil(n_samples / sps)) + 16

    if mod == "BPSK":
        bits = rng.integers(0, 2, n_symbols)
        symbols = (2 * bits - 1).astype(np.complex128)
    elif mod == "QPSK":
        bits = rng.integers(0, 2, n_symbols * 2)
        symbols = ((2 * bits[0::2] - 1) + 1j * (2 * bits[1::2] - 1)) / np.sqrt(2)
    elif mod == "8PSK":
        phases = rng.integers(0, 8, n_symbols) * (2 * np.pi / 8)
        symbols = np.exp(1j * phases)
    elif mod == "QAM16":
        grid = np.array([-3, -1, 1, 3])
        i_sym = rng.choice(grid, n_symbols)
        q_sym = rng.choice(grid, n_symbols)
        symbols = (i_sym + 1j * q_sym) / np.sqrt(10.0)
    elif mod == "QAM64":
        grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        i_sym = rng.choice(grid, n_symbols)
        q_sym = rng.choice(grid, n_symbols)
        symbols = (i_sym + 1j * q_sym) / np.sqrt(42.0)
    elif mod in ("GFSK", "CPFSK"):
        # Frequency modulation
        bits = rng.integers(0, 2, n_symbols)
        nrz = 2 * bits - 1
        upsampled = np.repeat(nrz, sps)
        if mod == "GFSK":
            # Gaussian smoothing
            L = sps * 3
            t_g = np.linspace(-1.5, 1.5, L)
            g_filter = np.exp(-2 * (np.pi * 0.5 * t_g) ** 2)
            g_filter /= np.sum(g_filter)
            freq_dev = np.convolve(upsampled, g_filter, mode="same")
        else:
            freq_dev = upsampled
        h = 0.5  # Modulation index
        phase = np.cumsum(freq_dev) * (np.pi * h / sps)
        raw_sig = np.exp(1j * phase)
        # Trim to length
        symbols = None
        sig = raw_sig[:n_samples]
    else:
        raise ValueError(f"Unknown mod {mod}")

    if symbols is not None:
        # Upsample and pulse shape
        upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
        upsampled[::sps] = symbols
        pulse = generate_pulse(sps=sps, beta=0.35, span=6)
        sig = np.convolve(upsampled, pulse, mode="same")[:n_samples]

    # Add Carrier Frequency Offset (CFO) and phase
    t = np.arange(len(sig))
    init_phase = rng.uniform(0, 2 * np.pi)
    sig = sig * np.exp(1j * (2 * np.pi * cfo_norm * t + init_phase))

    # Add AWGN
    sig_pwr = np.mean(np.abs(sig) ** 2)
    noise_pwr = sig_pwr / (10 ** (snr_db / 10.0))
    noise = (rng.normal(0, np.sqrt(noise_pwr / 2), len(sig)) +
             1j * rng.normal(0, np.sqrt(noise_pwr / 2), len(sig)))
    rx = sig + noise

    # Unit power normalize
    rx = rx / (np.sqrt(np.mean(np.abs(rx) ** 2)) + 1e-12)
    return rx.astype(np.complex64)


def build_dataset(samples_per_class: int = 400):
    print(f"Generating dataset: {samples_per_class} signals per class across 7 classes...")
    X = []
    y = []

    for cls_name in MODULATION_CLASSES:
        print(f"  Generating {cls_name}...")
        for _ in range(samples_per_class):
            snr = np.random.uniform(6.0, 28.0)
            cfo = np.random.uniform(-0.03, 0.03)
            sps = np.random.choice([4, 6, 8])
            n_samples = np.random.choice([256, 512, 1024])
            sig = generate_signal(cls_name, n_samples=n_samples, sps=sps, snr_db=snr, cfo_norm=cfo)
            feats = extract_24_features(sig)
            vec = feature_dict_to_vector(feats)
            X.append(vec)
            y.append(cls_name)

    return np.array(X, dtype=np.float32), np.array(y)


if __name__ == "__main__":
    X, y = build_dataset(samples_per_class=350)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    art_dir = os.path.join("backend", "modules", "module3", "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    model_path = os.path.join(art_dir, "rf_model.joblib")

    print("\nTraining Random Forest model with validated configuration...")
    rf_engine = RFEngine(model_path=model_path)
    model = rf_engine.train_and_save(X_train, y_train, save_path=model_path)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n==========================================")
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    print(f"==========================================\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=MODULATION_CLASSES))
    print(f"Model successfully saved to {model_path}!")
