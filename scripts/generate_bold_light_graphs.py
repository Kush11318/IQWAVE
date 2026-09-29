"""
Generate Separate, High-Contrast, Professional LIGHT MODE Graphs with BOLD Axes:
1. AMC Accuracy vs SNR (amc_accuracy_vs_snr_light.png)
2. Frame Detection Accuracy vs Channel BER (frame_detection_vs_ber_light.png)

Features:
- Pure Light Mode (Crisp White Background #FFFFFF)
- Extra Bold X and Y Axis Labels
- Extra Bold Tick Labels on Both X and Y Axes
- High-contrast, presentation-optimized typography and line weights
- Clear standalone figures (NOT joint)
- Flawless spacing with zero label collisions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate

os.makedirs('presentation_assets', exist_ok=True)

# Set global matplotlib font styling for bold, crisp academic presentation
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']

# ==============================================================================
# GRAPH 1: AMC ACCURACY VS SNR (LIGHT MODE - SEPARATE & BOLD)
# ==============================================================================

snr_db = np.array([-5, 0, 5, 10, 15, 20, 25, 30])
acc_fused = np.array([36.2, 48.5, 64.8, 79.2, 88.6, 93.4, 95.2, 96.0])

fig1, ax1 = plt.subplots(figsize=(8.8, 6.0), dpi=300)
fig1.patch.set_facecolor('#ffffff')
ax1.set_facecolor('#ffffff')

# Grid & Spines (Bold and Clean)
ax1.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.95, alpha=0.85, zorder=1)
for spine in ax1.spines.values():
    spine.set_color('#0f172a')
    spine.set_linewidth(2.0)

# EXTRA BOLD Ticks on X and Y
ax1.tick_params(colors='#0f172a', labelsize=12.0, length=7, width=2.0, direction='out')
for tick in ax1.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
for tick in ax1.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

# Colors
color_primary = '#0284c7'  # Deep Ocean Blue
color_benchmark = '#d97706'  # Amber / Dark Orange

# Shaded Area
ax1.fill_between(snr_db, acc_fused, 20, color=color_primary, alpha=0.10, zorder=2)

# Bold Main Curve
ax1.plot(snr_db, acc_fused, color=color_primary, lw=3.8, zorder=4,
         label='DAWC Dual-Branch Fused Engine (DBFCNN)')

# Markers
ax1.scatter(snr_db, acc_fused, color='#ffffff', edgecolor=color_primary,
            s=100, lw=2.8, zorder=5)

# Data Labels (Bold on Every Point with Clean Offset)
for x, y in zip(snr_db, acc_fused):
    if x == -5:
        offset_x, offset_y = 0.0, 3.8
    elif x == 0:
        offset_x, offset_y = 0.0, -5.4  # Placed cleanly below marker to avoid callout
    elif x == 5:
        offset_x, offset_y = 0.6, 4.2  # Slightly right/up to clear callout box
    elif x in [25, 30]:
        offset_x, offset_y = 0.0, 3.2
    else:
        offset_x, offset_y = 0.0, 3.8

    ax1.annotate(f"{y:.1f}%", xy=(x, y), xytext=(x + offset_x, y + offset_y),
                 ha='center', color='#0f172a', fontweight='bold', fontsize=10.0,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor='#ffffff', edgecolor='#cbd5e1', lw=1.2, alpha=0.96))

# Key Callouts (Strictly measured points distinguishing test regimes)
ax1.annotate('48.5% @ 0 dB\nSevere Rayleigh fading\n(Chance = 14.3%)',
             xy=(0, 48.5), xytext=(-6.4, 65.5),
             color='#b45309', fontweight='bold', fontsize=9.2,
             arrowprops=dict(facecolor='#b45309', edgecolor='#b45309', shrink=0.08, width=1.8, headwidth=6.0),
             bbox=dict(boxstyle="round,pad=0.35", facecolor='#fef3c7', edgecolor='#b45309', lw=1.5))

ax1.annotate('95.2% @ 25 dB\nLive multipath stress test',
             xy=(25, 95.2), xytext=(17.0, 83.0),
             color='#0369a1', fontweight='bold', fontsize=9.6,
             arrowprops=dict(facecolor='#0369a1', edgecolor='#0369a1', shrink=0.08, width=1.8, headwidth=6.0),
             bbox=dict(boxstyle="round,pad=0.4", facecolor='#e0f2fe', edgecolor='#0369a1', lw=1.5))

# Titles and BOLD Axis Labels (Exact technical specification)
ax1.set_title("AMC CLASSIFICATION ACCURACY VS. CHANNEL SNR\n"
              "Measured validation across noisy & multipath channel conditions | 7 classes",
              color='#0f172a', fontsize=13.6, fontweight='bold', pad=15)
ax1.set_xlabel("Channel Signal-to-Noise Ratio (SNR in dB)", color='#0f172a', fontsize=13.0, fontweight='bold', labelpad=10)
ax1.set_ylabel("Classification Accuracy (%)", color='#0f172a', fontsize=13.0, fontweight='bold', labelpad=10)

ax1.set_xlim(-6.8, 31.8)
ax1.set_ylim(22, 104)
ax1.legend(facecolor='#ffffff', edgecolor='#64748b', labelcolor='#0f172a', loc='lower right',
           fontsize=10.5, framealpha=0.98, fancybox=True).get_frame().set_linewidth(1.5)

plt.tight_layout()
g1_path = 'presentation_assets/amc_accuracy_vs_snr_light.png'
fig1.savefig(g1_path, facecolor='#ffffff', edgecolor='none', bbox_inches='tight')
plt.close(fig1)


# ==============================================================================
# GRAPH 2: FRAME DETECTION ACCURACY VS CHANNEL BER (LIGHT MODE - SEPARATE & BOLD)
# ==============================================================================

ber_points = np.array([0, 1, 5, 10, 20])
acc_frame_points = np.array([100.0, 99.86, 97.82, 89.84, 58.68])

fig2, ax2 = plt.subplots(figsize=(8.8, 6.0), dpi=300)
fig2.patch.set_facecolor('#ffffff')
ax2.set_facecolor('#ffffff')

# Grid & Spines (Bold and Clean)
ax2.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.95, alpha=0.85, zorder=1)
for spine in ax2.spines.values():
    spine.set_color('#0f172a')
    spine.set_linewidth(2.0)

# EXTRA BOLD Ticks on X and Y
ax2.tick_params(colors='#0f172a', labelsize=12.0, length=7, width=2.0, direction='out')
for tick in ax2.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
for tick in ax2.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

# Operational Zones
ax2.axvspan(0, 10, color='#10b981', alpha=0.07, zorder=2,
            label='Measured operating range in this benchmark')
ax2.axvspan(10, 21.5, color='#f43f5e', alpha=0.05, zorder=2)

# Smooth PCHIP curve
ber_fine = np.linspace(0, 20, 300)
acc_fine = pchip_interpolate(ber_points, acc_frame_points, ber_fine)

color_frame_curve = '#0284c7'  # Deep Ocean Blue
ax2.plot(ber_fine, acc_fine, color=color_frame_curve, lw=3.8, zorder=4)

# Actual Data Markers
ax2.scatter(ber_points, acc_frame_points, color='#ffffff', edgecolor='#0f172a',
            s=115, lw=2.8, zorder=5, label='Actual Measured Benchmark Data')

# Benchmark line at 89.84% @ 10% BER
ax2.axhline(89.84, color='#d97706', linestyle='--', lw=2.2, zorder=3,
            label='89.84% Benchmark @ 10% Channel BER')

# Point Annotations with smart offsets (completely uncollided)
ax2.annotate("100.00%", xy=(0, 100.0), xytext=(-0.35, 104.2),
             ha='right', color='#0f172a', fontweight='bold', fontsize=10.0,
             bbox=dict(boxstyle="round,pad=0.25", facecolor='#ffffff', edgecolor='#cbd5e1', lw=1.2, alpha=0.96))

ax2.annotate("99.86%", xy=(1, 99.86), xytext=(1.45, 104.2),
             ha='left', color='#0f172a', fontweight='bold', fontsize=10.0,
             bbox=dict(boxstyle="round,pad=0.25", facecolor='#ffffff', edgecolor='#cbd5e1', lw=1.2, alpha=0.96))

ax2.annotate("97.82%", xy=(5, 97.82), xytext=(5.0, 102.2),
             ha='center', color='#0f172a', fontweight='bold', fontsize=10.0,
             bbox=dict(boxstyle="round,pad=0.25", facecolor='#ffffff', edgecolor='#cbd5e1', lw=1.2, alpha=0.96))

ax2.annotate("89.84%", xy=(10, 89.84), xytext=(10.0, 94.2),
             ha='center', color='#0f172a', fontweight='bold', fontsize=10.0,
             bbox=dict(boxstyle="round,pad=0.25", facecolor='#ffffff', edgecolor='#cbd5e1', lw=1.2, alpha=0.96))

ax2.annotate("58.68%", xy=(20, 58.68), xytext=(20.0, 64.0),
             ha='center', color='#0f172a', fontweight='bold', fontsize=10.0,
             bbox=dict(boxstyle="round,pad=0.25", facecolor='#ffffff', edgecolor='#cbd5e1', lw=1.2, alpha=0.96))

# Explanatory callouts (Cleanly separated from data labels)
ax2.annotate('High Noise Resilience:\nMaintains 89.84% @ 10% BER\n(1 in 10 bits corrupted)',
             xy=(10, 89.84), xytext=(5.8, 71.0),
             color='#b45309', fontweight='bold', fontsize=9.6,
             arrowprops=dict(facecolor='#b45309', edgecolor='#b45309', shrink=0.08, width=1.8, headwidth=6.0),
             bbox=dict(boxstyle="round,pad=0.4", facecolor='#fef3c7', edgecolor='#b45309', lw=1.5))

ax2.annotate('Detection degrades as BER increases:\n58.68% @ 20% Channel BER',
             xy=(20, 58.68), xytext=(11.5, 43.0),
             color='#b91c1c', fontweight='bold', fontsize=9.4,
             arrowprops=dict(facecolor='#b91c1c', edgecolor='#b91c1c', shrink=0.08, width=1.8, headwidth=6.0),
             bbox=dict(boxstyle="round,pad=0.4", facecolor='#fee2e2', edgecolor='#b91c1c', lw=1.5))

# Titles and BOLD Axis Labels (Simplified for presentation clarity)
ax2.set_title("Frame Detection Accuracy vs. Channel BER",
              color='#0f172a', fontsize=15.0, fontweight='bold', pad=15)
ax2.set_xlabel("Channel Bit Error Rate (BER in %)", color='#0f172a', fontsize=13.0, fontweight='bold', labelpad=10)
ax2.set_ylabel("Detection Accuracy (%)", color='#0f172a', fontsize=13.0, fontweight='bold', labelpad=10)

ax2.set_xlim(-1.2, 21.2)
ax2.set_ylim(34, 109)
ax2.legend(facecolor='#ffffff', edgecolor='#64748b', labelcolor='#0f172a', loc='lower left',
           fontsize=10.2, framealpha=0.98, fancybox=True).get_frame().set_linewidth(1.5)

plt.tight_layout()
g2_path = 'presentation_assets/frame_detection_vs_ber_light.png'
fig2.savefig(g2_path, facecolor='#ffffff', edgecolor='none', bbox_inches='tight')
plt.close(fig2)

print("Successfully generated perfectly uncollided, bold, separate light-mode graphs in presentation_assets/!")
