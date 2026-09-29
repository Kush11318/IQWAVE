"""Generate Publication-Quality Professional AMC Graphs.

1. Confusion Matrix (Counts & Percentages + Marginals) for Dual-Branch Fused Engine
2. Top 10 Engineered Feature Importance Chart from Random Forest
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib

from backend.modules.module3.feature_extractor import FEATURE_NAMES_32, MODULATION_CLASSES

def generate_graphs():
    # Directories
    art_dir = os.path.join("backend", "modules", "module3", "artifacts")
    assets_dir = os.path.join("docs", "images")
    brain_dir = r"C:\Users\Dell\.gemini\antigravity-ide\brain\a8dd8184-259a-4c26-94e7-f02adf5165bb"
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(brain_dir, exist_ok=True)

    # Global Style Setup for Professional IEEE/Nature look
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#2c3e50"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.facecolor"] = "#ffffff"
    plt.rcParams["axes.facecolor"] = "#ffffff"

    # =========================================================================
    # 1. Confusion Matrix
    # =========================================================================
    cm = np.array([
        [347,  10,  14,   8,  18,   2,   1],
        [ 11, 265,  76,  18,  15,   9,   6],
        [ 16,  48, 294,  20,  15,   5,   2],
        [  6,  22,  19, 182, 166,   5,   0],
        [  5,   9,  17,  92, 267,   6,   4],
        [  3,   9,  14,   5,   5, 294,  70],
        [  6,   4,  12,  10,   2,  94, 272]
    ])
    classes = MODULATION_CLASSES
    n_classes = len(classes)
    row_sums = cm.sum(axis=1)
    cm_norm = cm.astype(np.float64) / row_sums[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

    # Custom colormap: crisp, modern blue gradient
    cmap = plt.cm.Blues
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Recall Ratio", rotation=-90, va="bottom", fontsize=11, fontweight="bold", color="#1a252f")
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold", color="#1a252f")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold", color="#1a252f")

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate counts and percentages in each cell
    thresh = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100.0
            color = "white" if cm_norm[i, j] > thresh else "#1a252f"
            text = f"{val}\n({pct:.1f}%)"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color=color,
                    fontsize=9.5,
                    fontweight="bold" if i == j else "normal")

    ax.set_title("IQWAVE Dual-Branch Fused Engine — Confusion Matrix\n"
                 r"Evaluated across SNR $\in [-5, +30\text{ dB}]$, Rayleigh Fading & IQ Imbalance (N = 2,800)",
                 fontsize=12, fontweight="bold", pad=16, color="#0f172a")
    ax.set_ylabel("True Modulation Class", fontsize=12, fontweight="bold", labelpad=10, color="#1e293b")
    ax.set_xlabel("Predicted Modulation Class", fontsize=12, fontweight="bold", labelpad=10, color="#1e293b")

    # Subtle grid
    ax.set_xticks(np.arange(n_classes + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_classes + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#cbd5e1", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    cm_path_assets = os.path.join(assets_dir, "amc_confusion_matrix.png")
    cm_path_brain = os.path.join(brain_dir, "amc_confusion_matrix.png")
    fig.savefig(cm_path_assets, dpi=300, bbox_inches="tight")
    fig.savefig(cm_path_brain, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion Matrix saved to:\n  - {cm_path_assets}\n  - {cm_path_brain}")

    # =========================================================================
    # 2. Top 10 Feature Importance Chart
    # =========================================================================
    rf_path = os.path.join(art_dir, "rf_model.joblib")
    rf_model = joblib.load(rf_path)
    importances = rf_model.feature_importances_

    sorted_idx = np.argsort(importances)[::-1][:10]
    top_names = [FEATURE_NAMES_32[i] for i in sorted_idx]
    top_values = [importances[i] * 100.0 for i in sorted_idx]

    # Friendly labels with domain categories
    feature_meta = {
        "phase_m2_concentration": ("Phase M2 Concentration", "Phase (BPSK vs Multi-Phase)", "#2563eb"),
        "phase_m4_concentration": ("Phase M4 Concentration", "Phase (QPSK 4-Phasor)", "#3b82f6"),
        "amp_p10": ("Amplitude 10th Percentile", "Envelope Dynamics (QAM Lower Tail)", "#0284c7"),
        "amp_skewness": ("Amplitude Skewness", "Envelope Asymmetry", "#0ea5e9"),
        "c40_norm": ("Fourth-Order Cumulant C40", "Higher-Order Statistics", "#10b981"),
        "amp_mean": ("Mean Envelope Amplitude", "First-Order Envelope Stat", "#059669"),
        "amp_cv": ("Amplitude Coeff. of Variation", "Constellation Spread Ratio", "#047857"),
        "amp_std": ("Amplitude Standard Deviation", "Constellation Power Dispersion", "#14b8a6"),
        "spectral_flatness": ("Spectral Flatness (Wiener)", "Frequency Domain (Wiener Entropy)", "#f59e0b"),
        "gated_phase_diff_std": ("Gated Phase Diff Std", "Phase Stability / SNR Gated", "#6366f1")
    }

    friendly_labels = [feature_meta.get(f, (f, "", "#2563eb"))[0] for f in top_names]
    bar_colors = [feature_meta.get(f, (f, "", "#2563eb"))[2] for f in top_names]

    # Invert so highest is at top
    top_names_rev = friendly_labels[::-1]
    top_values_rev = top_values[::-1]
    bar_colors_rev = bar_colors[::-1]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    y_pos = np.arange(len(top_names_rev))
    bars = ax.barh(y_pos, top_values_rev, height=0.62, color=bar_colors_rev, edgecolor="#1e293b", linewidth=0.8, alpha=0.92)

    # Gridlines
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.set_axisbelow(True)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.15, bar.get_y() + bar.get_height() / 2.0,
                f"{width:.2f}%",
                va="center", ha="left",
                fontsize=10, fontweight="bold", color="#0f172a")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names_rev, fontsize=10.5, fontweight="bold", color="#1e293b")
    ax.set_xlabel("Relative Gini Feature Importance (%)", fontsize=11, fontweight="bold", labelpad=8, color="#0f172a")
    ax.set_xlim(0, max(top_values) * 1.18)

    ax.set_title("IQWAVE Random Forest Engine — Top 10 Discriminative Features\n"
                 "Extracted from 32 Handcrafted Features trained on 14,000 Signals (300 Trees)",
                 fontsize=12, fontweight="bold", pad=14, color="#0f172a")

    # Add subtitle legend / note
    total_top10 = sum(top_values)
    ax.text(0.98, 0.04, f"Top 10 Cumulative Importance: {total_top10:.1f}%\nTotal Feature Space: 32 Features",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9))

    plt.tight_layout()

    feat_path_assets = os.path.join(assets_dir, "amc_top_10_features.png")
    feat_path_brain = os.path.join(brain_dir, "amc_top_10_features.png")
    fig.savefig(feat_path_assets, dpi=300, bbox_inches="tight")
    fig.savefig(feat_path_brain, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Top 10 Features chart saved to:\n  - {feat_path_assets}\n  - {feat_path_brain}")


if __name__ == "__main__":
    generate_graphs()
