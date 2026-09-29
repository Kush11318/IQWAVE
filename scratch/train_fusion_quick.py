import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from backend.modules.module3.feature_extractor import MODULATION_CLASSES, FEATURE_NAMES_32, extract_features, feature_dict_to_vector
from backend.modules.module3.preprocessing import preprocess_for_cnn
from backend.modules.module3.multiscale_cnn_engine import MultiScaleDilatedCNN
from backend.modules.module3.fusion_engine import DualBranchFusionHead
from scratch.train_v2 import generate_signal

def quick_train_fusion():
    art_dir = os.path.join("backend", "modules", "module3", "artifacts")
    cnn_path = os.path.join(art_dir, "ensemble_cnn_0.pt")
    fusion_path = os.path.join(art_dir, "fusion_head.pt")

    if not os.path.exists(cnn_path):
        print("ensemble_cnn_0.pt not found!")
        return

    device = torch.device("cpu")
    cnn = MultiScaleDilatedCNN(num_classes=7)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
    cnn.eval()

    samples_per_class = 200
    print(f"Generating {samples_per_class * 7} signals for fusion head training...")

    X_rf = []
    X_gap = []
    y_idx = []

    for c_i, cls in enumerate(MODULATION_CLASSES):
        for _ in range(samples_per_class):
            snr = float(np.random.uniform(-5.0, 30.0))
            sig = generate_signal(cls, snr_db=snr, with_fading=True, with_iq_imbalance=True)

            feats = extract_features(sig)
            rf_vec = feature_dict_to_vector(feats, feature_names=FEATURE_NAMES_32)
            X_rf.append(rf_vec)

            cnn_slice, _ = preprocess_for_cnn(sig, target_length=128)
            with torch.no_grad():
                t_in = torch.from_numpy(cnn_slice).unsqueeze(0)
                gap_vec = cnn.extract_gap_features(t_in)[0].numpy()
            X_gap.append(gap_vec)

            y_idx.append(c_i)

    X_rf = np.array(X_rf, dtype=np.float32)
    X_gap = np.array(X_gap, dtype=np.float32)
    y_idx = np.array(y_idx, dtype=np.int64)

    idx_train, idx_test = train_test_split(np.arange(len(y_idx)), test_size=0.2, random_state=42, stratify=y_idx)

    train_ds = TensorDataset(torch.from_numpy(X_rf[idx_train]), torch.from_numpy(X_gap[idx_train]), torch.from_numpy(y_idx[idx_train]))
    test_ds = TensorDataset(torch.from_numpy(X_rf[idx_test]), torch.from_numpy(X_gap[idx_test]), torch.from_numpy(y_idx[idx_test]))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    head = DualBranchFusionHead(num_classes=7, rf_dim=32, cnn_dim=256)
    opt = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, 16):
        head.train()
        for brf, bcnn, by in train_loader:
            opt.zero_grad()
            out = head(brf, bcnn)
            loss = crit(out, by)
            loss.backward()
            opt.step()

        if ep % 5 == 0 or ep == 15:
            head.eval()
            corr = 0
            tot = 0
            with torch.no_grad():
                for brf, bcnn, by in test_loader:
                    out = head(brf, bcnn)
                    corr += (out.argmax(dim=-1) == by).sum().item()
                    tot += len(by)
            print(f"Epoch {ep:2d} | Val Acc: {corr/tot*100:.2f}%")

    torch.save(head.state_dict(), fusion_path)
    print(f"Fusion head saved to {fusion_path}!")

if __name__ == "__main__":
    quick_train_fusion()
