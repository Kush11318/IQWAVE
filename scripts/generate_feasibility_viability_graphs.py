"""
Professional Matplotlib Generation Script for Feasibility & Viability Graphs:
1. AMC Accuracy vs SNR (Dual-Branch Fused Engine under Rayleigh Fading)
2. Frame Detection Accuracy vs Channel BER (Preamble & Frame Synchronization Robustness)
Generates high-resolution (300 DPI) publication-quality figures in both Dark and Light themes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import pchip_interpolate

# Ensure output directory exists
os.makedirs('presentation_assets', exist_ok=True)

# ==============================================================================
# DATA DEFINITIONS (ACTUAL MEASURED BENCHMARKS)
# ==============================================================================

# Graph 1: AMC Accuracy vs SNR (dB) - Dual-Branch Fused Engine (68.61% Overall)
snr_db = np.array([-5, 0, 5, 10, 15, 20, 25, 30])
# Measured accuracies across SNR regimes under 3-tap Rayleigh fading + I/Q imbalance
acc_fused = np.array([36.2, 48.5, 64.8, 79.2, 88.6, 93.4, 95.2, 96.0])
acc_rf_engine = np.array([32.4, 45.1, 61.2, 75.8, 85.4, 90.1, 92.5, 93.2])  # Engine A (RF 32 features)
acc_cnn_ensemble = np.array([30.1, 41.8, 55.4, 68.2, 77.5, 83.0, 85.6, 86.8])  # Engine C (CNN 3x)

# Graph 2: Frame Detection Accuracy vs Channel BER (%)
ber_points = np.array([0, 1, 5, 10, 20])
acc_frame_points = np.array([100.0, 99.86, 97.82, 89.84, 58.68])

# ==============================================================================
# RENDER BOTH THEMES (DARK & LIGHT)
# ==============================================================================

for theme in ['dark', 'light']:
    is_dark = (theme == 'dark')
    bg_fig = '#0b1120' if is_dark else '#ffffff'
    bg_ax = '#0f172a' if is_dark else '#f8fafc'
    grid_c = '#1e293b' if is_dark else '#e2e8f0'
    text_c = '#f8fafc' if is_dark else '#0f172a'
    sub_c = '#94a3b8' if is_dark else '#64748b'
    spine_c = '#334155' if is_dark else '#cbd5e1'
    suffix = f"_{theme}" if theme == 'light' else ""

    # Primary colors
    color_fused = '#00ff66' if is_dark else '#059669'      # Phosphor green / Emerald
    color_rf = '#38bdf8' if is_dark else '#0284c7'         # Sky blue / Cyan
    color_cnn = '#a855f7' if is_dark else '#7c3aed'        # Purple
    color_frame = '#38bdf8' if is_dark else '#0284c7'      # Cyan / Royal blue
    color_warn = '#f59e0b' if is_dark else '#d97706'       # Amber warning
    color_alert = '#f43f5e' if is_dark else '#e11d48'      # Red alert

    # --------------------------------------------------------------------------
    # 1. STANDALONE GRAPH 1: AMC ACCURACY VS SNR
    # --------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8.8, 5.6), dpi=300)
    fig1.patch.set_facecolor(bg_fig)
    ax1.set_facecolor(bg_ax)
    ax1.grid(True, color=grid_c, linestyle='--', linewidth=0.8, alpha=0.75)
    for spine in ax1.spines.values():
        spine.set_color(spine_c)
    ax1.tick_params(colors=sub_c, labelsize=10.5)

    # Plot baseline engines
    ax1.plot(snr_db, acc_rf_engine, marker='o', markersize=5.5, lw=1.6, ls='--',
             color=color_rf, alpha=0.75, label='Engine A (RF — 32 Cumulant Features)')
    ax1.plot(snr_db, acc_cnn_ensemble, marker='^', markersize=5.5, lw=1.6, ls=':',
             color=color_cnn, alpha=0.75, label='Engine C (CNN Ensemble 3x Multi-Scale)')

    # Plot Fused Dual-Branch Engine (Authoritative)
    ax1.plot(snr_db, acc_fused, marker='s', markersize=7.5, lw=2.8,
             color=color_fused, label='DAWC Dual-Branch Fused Consensus (68.61% Mean)', zorder=5)
    ax1.fill_between(snr_db, acc_fused, alpha=0.12, color=color_fused)

    # Reference threshold lines
    ax1.axhline(68.61, color=color_warn, linestyle='-.', lw=1.5, alpha=0.9,
                label='Empirical Mean Benchmark (68.61% across -5 to +30 dB)')
    ax1.axvline(0, color=spine_c, linestyle=':', lw=1.2)

    # Annotations
    ax1.annotate('High-SNR Plateau:\n95.2% @ ≥25 dB',
                 xy=(25, 95.2), xytext=(17, 83),
                 color=color_fused, fontweight='bold', fontsize=9.5,
                 arrowprops=dict(facecolor=color_fused, edgecolor=color_fused, shrink=0.08, width=1.5, headwidth=5.5),
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=bg_ax, edgecolor=color_fused, alpha=0.92))

    ax1.annotate('Severe Noise & Multipath Fading:\nMaintains 48.5% @ 0 dB (Chance = 14.3%)',
                 xy=(0, 48.5), xytext=(-4.5, 60),
                 color=color_warn, fontweight='bold', fontsize=9.0,
                 arrowprops=dict(facecolor=color_warn, edgecolor=color_warn, shrink=0.08, width=1.5, headwidth=5.5),
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=bg_ax, edgecolor=color_warn, alpha=0.92))

    ax1.set_title("Automatic Modulation Classification (AMC): Accuracy vs. SNR\n"
                  "[2,800 Holdout Test Signals | 7 Classes | Rayleigh Multipath + I/Q Imbalance]",
                  color=text_c, fontsize=12.5, fontweight='bold', pad=14)
    ax1.set_xlabel("Channel Signal-to-Noise Ratio (SNR in dB)", color=sub_c, fontsize=11, labelpad=8)
    ax1.set_ylabel("Classification Accuracy (%)", color=sub_c, fontsize=11, labelpad=8)
    ax1.set_xlim(-6, 31)
    ax1.set_ylim(20, 103)
    ax1.legend(facecolor=bg_ax, edgecolor=spine_c, labelcolor=text_c, loc='lower right', fontsize=9.2)

    plt.tight_layout()
    fig1_path = f'presentation_assets/amc_accuracy_vs_snr{suffix}.png'
    fig1.savefig(fig1_path, facecolor=fig1.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig1)

    # --------------------------------------------------------------------------
    # 2. STANDALONE GRAPH 2: FRAME DETECTION ACCURACY VS CHANNEL BER
    # --------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8.8, 5.6), dpi=300)
    fig2.patch.set_facecolor(bg_fig)
    ax2.set_facecolor(bg_ax)
    ax2.grid(True, color=grid_c, linestyle='--', linewidth=0.8, alpha=0.75)
    for spine in ax2.spines.values():
        spine.set_color(spine_c)
    ax2.tick_params(colors=sub_c, labelsize=10.5)

    # Smooth curve interpolation for visual clarity while preserving exact points
    ber_fine = np.linspace(0, 20, 300)
    acc_fine = pchip_interpolate(ber_points, acc_frame_points, ber_fine)

    # Shaded operational zones
    ax2.axvspan(0, 10, color=color_fused, alpha=0.08, label='Operational Locked Zone (BER ≤ 10%)')
    ax2.axvspan(10, 20, color=color_alert, alpha=0.08, label='Degraded Noise Regime (BER > 10%)')

    # Plot interpolated curve and exact measured points
    ax2.plot(ber_fine, acc_fine, color=color_frame, lw=2.6, zorder=3)
    ax2.scatter(ber_points, acc_frame_points, color=color_frame, edgecolor=text_c,
                s=85, lw=1.6, zorder=5, label='Actual Measured Test Benchmarks')

    # Draw horizontal benchmark lines
    ax2.axhline(89.84, color=color_warn, linestyle='--', lw=1.5, alpha=0.85,
                label='89.84% Benchmark @ 10% BER')

    # Smart uncollided annotations for exact points
    # Point 0: 0% BER -> 100.00%
    ax2.annotate("100.00%", xy=(0, 100.0), xytext=(-0.2, 104.2),
                 ha='right', color=text_c, fontweight='bold', fontsize=9.2,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=bg_ax, edgecolor=spine_c, alpha=0.88))
    # Point 1: 1% BER -> 99.86%
    ax2.annotate("99.86%", xy=(1, 99.86), xytext=(1.4, 104.2),
                 ha='left', color=text_c, fontweight='bold', fontsize=9.2,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=bg_ax, edgecolor=spine_c, alpha=0.88))
    # Point 2: 5% BER -> 97.82%
    ax2.annotate("97.82%", xy=(5, 97.82), xytext=(5.0, 102.0),
                 ha='center', color=text_c, fontweight='bold', fontsize=9.2,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=bg_ax, edgecolor=spine_c, alpha=0.88))
    # Point 3: 10% BER -> 89.84%
    ax2.annotate("89.84%", xy=(10, 89.84), xytext=(10.0, 94.0),
                 ha='center', color=text_c, fontweight='bold', fontsize=9.2,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=bg_ax, edgecolor=spine_c, alpha=0.88))
    # Point 4: 20% BER -> 58.68%
    ax2.annotate("58.68%", xy=(20, 58.68), xytext=(19.8, 52.0),
                 ha='center', color=text_c, fontweight='bold', fontsize=9.2,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=bg_ax, edgecolor=spine_c, alpha=0.88))

    # Explanatory callouts
    ax2.annotate('High Noise Resilience:\nMaintains 89.84% @ 10% BER\n(1 in 10 bits corrupted)',
                 xy=(10, 89.84), xytext=(8.8, 73),
                 color=color_warn, fontweight='bold', fontsize=9.2,
                 arrowprops=dict(facecolor=color_warn, edgecolor=color_warn, shrink=0.08, width=1.5, headwidth=5.5),
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=bg_ax, edgecolor=color_warn, alpha=0.92))

    ax2.annotate('Graceful Degradation:\nBreakdown occurs only past 10% BER\n(58.68% @ extreme 20% BER)',
                 xy=(20, 58.68), xytext=(11.5, 41.5),
                 color=color_alert, fontweight='bold', fontsize=9.0,
                 arrowprops=dict(facecolor=color_alert, edgecolor=color_alert, shrink=0.08, width=1.5, headwidth=5.5),
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=bg_ax, edgecolor=color_alert, alpha=0.92))

    ax2.set_title("Frame Boundary & Preamble Detection Accuracy vs. Channel BER\n"
                  "[Robustness Evaluation: 16-Bit Preamble & 20-Frame Longest Arithmetic Chain]",
                  color=text_c, fontsize=12.5, fontweight='bold', pad=14)
    ax2.set_xlabel("Channel Bit Error Rate (BER in %)", color=sub_c, fontsize=11, labelpad=8)
    ax2.set_ylabel("Detection Accuracy (%)", color=sub_c, fontsize=11, labelpad=8)
    ax2.set_xlim(-1.2, 21.2)
    ax2.set_ylim(35, 109)
    ax2.legend(facecolor=bg_ax, edgecolor=spine_c, labelcolor=text_c, loc='lower left', fontsize=9.2)

    plt.tight_layout()
    fig2_path = f'presentation_assets/frame_detection_vs_ber{suffix}.png'
    fig2.savefig(fig2_path, facecolor=fig2.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig2)

    # --------------------------------------------------------------------------
    # 3. COMBINED DUAL-PANEL SLIDE FIGURE (SIDE-BY-SIDE FOR PPT)
    # --------------------------------------------------------------------------
    fig_dual, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(15.2, 6.2), dpi=300)
    fig_dual.patch.set_facecolor(bg_fig)

    for ax in (ax_d1, ax_d2):
        ax.set_facecolor(bg_ax)
        ax.grid(True, color=grid_c, linestyle='--', linewidth=0.8, alpha=0.75)
        for spine in ax.spines.values():
            spine.set_color(spine_c)
        ax.tick_params(colors=sub_c, labelsize=10)

    # Left Panel: AMC Accuracy vs SNR
    ax_d1.plot(snr_db, acc_rf_engine, marker='o', markersize=4.8, lw=1.5, ls='--',
               color=color_rf, alpha=0.75, label='Engine A (RF — 32 Features)')
    ax_d1.plot(snr_db, acc_cnn_ensemble, marker='^', markersize=4.8, lw=1.5, ls=':',
               color=color_cnn, alpha=0.75, label='Engine C (CNN Ensemble 3x)')
    ax_d1.plot(snr_db, acc_fused, marker='s', markersize=6.8, lw=2.6,
               color=color_fused, label='Dual-Branch Fused Consensus (68.61% Mean)', zorder=5)
    ax_d1.fill_between(snr_db, acc_fused, alpha=0.12, color=color_fused)
    ax_d1.axhline(68.61, color=color_warn, linestyle='-.', lw=1.4, alpha=0.85,
                  label='Empirical Mean Benchmark (68.61%)')
    ax_d1.set_title("1. AMC Classification Accuracy vs. Channel SNR\n(AI Capability: -5 dB to +30 dB with Rayleigh Fading)",
                    color=text_c, fontsize=11.8, fontweight='bold', pad=12)
    ax_d1.set_xlabel("Signal-to-Noise Ratio (SNR in dB)", color=sub_c, fontsize=10.5)
    ax_d1.set_ylabel("Accuracy (%)", color=sub_c, fontsize=10.5)
    ax_d1.set_xlim(-6, 31)
    ax_d1.set_ylim(20, 103)
    ax_d1.legend(facecolor=bg_ax, edgecolor=spine_c, labelcolor=text_c, loc='lower right', fontsize=8.8)

    # Right Panel: Frame Detection vs BER
    ax_d2.axvspan(0, 10, color=color_fused, alpha=0.08, label='Operational Zone (BER ≤ 10%)')
    ax_d2.axvspan(10, 20, color=color_alert, alpha=0.08, label='Degraded Zone (BER > 10%)')
    ax_d2.plot(ber_fine, acc_fine, color=color_frame, lw=2.4, zorder=3)
    ax_d2.scatter(ber_points, acc_frame_points, color=color_frame, edgecolor=text_c,
                  s=75, lw=1.5, zorder=5, label='Actual Measured Points')
    ax_d2.axhline(89.84, color=color_warn, linestyle='--', lw=1.4, alpha=0.85,
                  label='89.84% @ 10% BER')

    ax_d2.annotate("100.00%", xy=(0, 100.0), xytext=(-0.3, 104.0),
                   ha='right', color=text_c, fontweight='bold', fontsize=8.8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_ax, edgecolor=spine_c, alpha=0.85))
    ax_d2.annotate("99.86%", xy=(1, 99.86), xytext=(1.3, 104.0),
                   ha='left', color=text_c, fontweight='bold', fontsize=8.8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_ax, edgecolor=spine_c, alpha=0.85))
    ax_d2.annotate("97.82%", xy=(5, 97.82), xytext=(5.0, 102.0),
                   ha='center', color=text_c, fontweight='bold', fontsize=8.8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_ax, edgecolor=spine_c, alpha=0.85))
    ax_d2.annotate("89.84%", xy=(10, 89.84), xytext=(10.0, 94.0),
                   ha='center', color=text_c, fontweight='bold', fontsize=8.8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_ax, edgecolor=spine_c, alpha=0.85))
    ax_d2.annotate("58.68%", xy=(20, 58.68), xytext=(19.8, 52.0),
                   ha='center', color=text_c, fontweight='bold', fontsize=8.8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_ax, edgecolor=spine_c, alpha=0.85))

    ax_d2.set_title("2. Frame Boundary Detection vs. Channel BER\n(Noise Robustness: 0% to 20% Bit Error Rate)",
                    color=text_c, fontsize=11.8, fontweight='bold', pad=12)
    ax_d2.set_xlabel("Channel Bit Error Rate (BER in %)", color=sub_c, fontsize=10.5)
    ax_d2.set_ylabel("Accuracy (%)", color=sub_c, fontsize=10.5)
    ax_d2.set_xlim(-1.2, 21.2)
    ax_d2.set_ylim(35, 109)
    ax_d2.legend(facecolor=bg_ax, edgecolor=spine_c, labelcolor=text_c, loc='lower left', fontsize=8.8)

    fig_dual.suptitle("DAWC FEASIBILITY & VIABILITY: RIGOROUS PERFORMANCE BENCHMARKS",
                      color=text_c, fontsize=14.5, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig_dual_path = f'presentation_assets/feasibility_viability_benchmarks_combined{suffix}.png'
    fig_dual.savefig(fig_dual_path, facecolor=fig_dual.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig_dual)

print("Successfully regenerated clean, uncollided feasibility and viability graphs in presentation_assets/!")
