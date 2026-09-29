import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from backend.modules.module3.feature_extractor import (
    FEATURE_NAMES_24,
    MODULATION_CLASSES,
    extract_24_features,
    feature_dict_to_vector
)
from backend.modules.module3.preprocessing import preprocess_for_cnn
from backend.modules.module3.rf_engine import RFEngine
from backend.modules.module3.cnn_engine import RawIQCNN


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
    """Synthesize high-fidelity baseband I/Q signal."""
    rng = np.random.default_rng()
    n_symbols = int(np.ceil(n_samples / sps)) + 24

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
        bits = rng.integers(0, 2, n_symbols)
        nrz = 2 * bits - 1
        upsampled = np.repeat(nrz, sps)
        if mod == "GFSK":
            L = sps * 3
            t_g = np.linspace(-1.5, 1.5, L)
            g_filter = np.exp(-2 * (np.pi * 0.5 * t_g) ** 2)
            g_filter /= np.sum(g_filter)
            freq_dev = np.convolve(upsampled, g_filter, mode="same")
        else:
            freq_dev = upsampled
        h = 0.5
        phase = np.cumsum(freq_dev) * (np.pi * h / sps)
        raw_sig = np.exp(1j * phase)
        symbols = None
        sig = raw_sig[:n_samples]
    else:
        raise ValueError(f"Unknown mod {mod}")

    if symbols is not None:
        upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
        upsampled[::sps] = symbols
        pulse = generate_pulse(sps=sps, beta=rng.uniform(0.25, 0.45), span=8)
        sig = np.convolve(upsampled, pulse, mode="same")[:n_samples]

    t = np.arange(len(sig))
    init_phase = rng.uniform(0, 2 * np.pi)
    sig = sig * np.exp(1j * (2 * np.pi * cfo_norm * t + init_phase))

    sig_pwr = np.mean(np.abs(sig) ** 2)
    noise_pwr = sig_pwr / (10 ** (snr_db / 10.0))
    noise = (rng.normal(0, np.sqrt(noise_pwr / 2), len(sig)) +
             1j * rng.normal(0, np.sqrt(noise_pwr / 2), len(sig)))
    rx = sig + noise

    rx = rx / (np.sqrt(np.mean(np.abs(rx) ** 2)) + 1e-12)
    return rx.astype(np.complex64)


def build_dual_datasets(samples_per_class: int = 500):
    print(f"Synthesizing {samples_per_class * 7} signals across 7 classes...")
    X_rf = []
    X_cnn = []
    y_labels = []

    for cls_idx, cls_name in enumerate(MODULATION_CLASSES):
        print(f"  Generating {cls_name}...")
        for _ in range(samples_per_class):
            snr = np.random.uniform(5.0, 30.0)
            cfo = np.random.uniform(-0.04, 0.04)
            sps = int(np.random.choice([4, 6, 8]))
            n_samples = int(np.random.choice([256, 512, 1024]))

            sig = generate_signal(cls_name, n_samples=n_samples, sps=sps, snr_db=snr, cfo_norm=cfo)

            # Feature vector for Random Forest
            feats = extract_24_features(sig)
            vec = feature_dict_to_vector(feats)
            X_rf.append(vec)

            # 128x2 RMS-normalized slice for CNN
            cnn_slice, _ = preprocess_for_cnn(sig, target_length=128)
            X_cnn.append(cnn_slice)

            y_labels.append(cls_name)

    return (
        np.array(X_rf, dtype=np.float32),
        np.array(X_cnn, dtype=np.float32),
        np.array(y_labels)
    )


def train_cnn(X_train, y_train_idx, X_test, y_test_idx, epochs=25, batch_size=64):
    device = torch.device("cpu")
    model = RawIQCNN(num_classes=7).to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_idx))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test_idx))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\nTraining RawIQCNN (Engine B)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            preds = out.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += len(by)

        scheduler.step()
        train_acc = correct / total

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    val_correct += (out.argmax(dim=-1) == by).sum().item()
                    val_total += len(by)
            val_acc = val_correct / val_total
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    return model


if __name__ == "__main__":
    X_rf, X_cnn, y_str = build_dual_datasets(samples_per_class=450)
    print(f"Dataset summary: X_rf={X_rf.shape}, X_cnn={X_cnn.shape}, y={y_str.shape}")

    class_to_idx = {name: i for i, name in enumerate(MODULATION_CLASSES)}
    y_idx = np.array([class_to_idx[s] for s in y_str], dtype=np.int64)

    # Train / Test split
    indices = np.arange(len(y_str))
    idx_train, idx_test = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_idx)

    # =========================================================================
    # 1. Train & Save Engine A (Random Forest)
    # =========================================================================
    art_dir = os.path.join("backend", "modules", "module3", "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    rf_path = os.path.join(art_dir, "rf_model.joblib")

    print("\n" + "=" * 50)
    print("Training Engine A: 24-Feature Random Forest...")
    print("=" * 50)
    rf_engine = RFEngine(model_path=rf_path)
    rf_model = rf_engine.train_and_save(X_rf[idx_train], y_str[idx_train], save_path=rf_path)

    rf_pred = rf_model.predict(X_rf[idx_test])
    rf_acc = accuracy_score(y_str[idx_test], rf_pred)
    print(f"\nEngine A (Random Forest) Validation Accuracy: {rf_acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_str[idx_test], rf_pred, target_names=MODULATION_CLASSES))

    # =========================================================================
    # 2. Train & Save Engine B (PyTorch 1D CNN)
    # =========================================================================
    cnn_path = os.path.join(art_dir, "cnn_model.pt")
    print("\n" + "=" * 50)
    print("Training Engine B: 101,319-Parameter 1D Raw-IQ CNN...")
    print("=" * 50)
    cnn_model = train_cnn(
        X_cnn[idx_train], y_idx[idx_train],
        X_cnn[idx_test], y_idx[idx_test],
        epochs=22, batch_size=64
    )

    torch.save(cnn_model.state_dict(), cnn_path)
    print(f"Engine B weights saved successfully to {cnn_path}!")

    # Test CNN accuracy
    cnn_model.eval()
    with torch.no_grad():
        test_in = torch.from_numpy(X_cnn[idx_test])
        cnn_probs = cnn_model(test_in).numpy()
        cnn_preds = [MODULATION_CLASSES[i] for i in cnn_probs.argmax(axis=-1)]
    cnn_acc = accuracy_score(y_str[idx_test], cnn_preds)
    print(f"\nEngine B (Raw-IQ CNN) Validation Accuracy: {cnn_acc * 100:.2f}%")

    # =========================================================================
    # 3. Dual Engine Ensemble Evaluation
    # =========================================================================
    rf_probs = rf_model.predict_proba(X_rf[idx_test])
    ensemble_probs = 0.55 * rf_probs + 0.45 * cnn_probs
    ensemble_preds = [MODULATION_CLASSES[i] for i in ensemble_probs.argmax(axis=-1)]
    ens_acc = accuracy_score(y_str[idx_test], ensemble_preds)

    print("\n" + "=" * 50)
    print(f"DUAL-ENGINE ENSEMBLE ACCURACY: {ens_acc * 100:.2f}%")
    print(f"Improvement over Single Engine: +{max(0.0, ens_acc - max(rf_acc, cnn_acc)) * 100:.2f}%")
    print("=" * 50)
    print("\nEnsemble Classification Report:")
    print(classification_report(y_str[idx_test], ensemble_preds, target_names=MODULATION_CLASSES))
