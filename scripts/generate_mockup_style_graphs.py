import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

# Publication-grade typography
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
rcParams['mathtext.fontset'] = 'dejavusans'
rcParams['axes.edgecolor'] = '#334e68'
rcParams['axes.linewidth'] = 1.3

def plot_top10_features_compact():
    # Concise, punchy, judge-friendly names that fit cleanly on presentation slides
    features_data = [
        (r"$\mathbf{BPSK\ Phase\ Coherence}\ \Psi_{2}$", 7.70),
        (r"$\mathbf{QPSK\ Phase\ Coherence}\ \Psi_{4}$", 6.23),
        (r"$\mathbf{Amplitude\ Floor}\ P_{10}$", 5.37),
        (r"$\mathbf{Envelope\ Skewness}\ \gamma_{1}$", 4.56),
        (r"$\mathbf{4th\ Cumulant}\ |C_{40}|$", 4.47),
        (r"$\mathbf{Mean\ Amplitude}\ \mu_{a}$", 3.97),
        (r"$\mathbf{Envelope\ Fluctuation}\ CV_{a}$", 3.95),
        (r"$\mathbf{Envelope\ Spread}\ \sigma_{a}$", 3.80),
        (r"$\mathbf{Spectral\ Flatness}\ \Xi_{f}$", 3.77),
        (r"$\mathbf{Phase\ Jitter}\ \sigma_{\Delta\theta}$", 3.64),
    ]

    labels = [item[0] for item in reversed(features_data)]
    values = [item[1] for item in reversed(features_data)]
    y_pos = np.arange(len(labels))

    # Compact slide-friendly figure proportions (8.8 x 5.4)
    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=300)

    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f6faff')

    # Gridlines
    ax.set_axisbelow(True)
    ax.grid(axis='x', linestyle='--', linewidth=1.0, color='#cde2f2', dashes=(4, 4), alpha=0.95)
    ax.grid(axis='y', visible=False)

    # Bars
    bar_color = '#1785cc'
    edge_color = '#116ca8'
    bars = ax.barh(y_pos, values, height=0.56, color=bar_color, edgecolor=edge_color, linewidth=1.0)

    # Y-axis labels - compact & readable
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5, fontweight='bold', color='#102a43')
    ax.tick_params(axis='y', length=0, pad=8)

    # X-axis ticks in direct %
    ax.set_xlim(0.0, 9.0)
    x_ticks = np.arange(0.0, 9.5, 1.0)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0, symbol='%'))
    ax.tick_params(axis='x', labelsize=10.5, colors='#102a43', width=1.2, length=4.5)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')

    # Percentage annotations at end of bars
    for bar, val in zip(bars, values):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2.0
        ax.text(w + 0.14, y, f"{val:.1f}%",
                va='center', ha='left',
                fontsize=10.5, fontweight='bold', color='#102a43')

    # Short, compact X-axis label
    ax.set_xlabel('Feature Importance (%)',
                  fontsize=11.5, fontweight='bold', color='#102a43', labelpad=9)

    # Title
    ax.set_title('Module 3 AMC: Top-10 Handcrafted DSP Feature Importances',
                 fontsize=13.0, fontweight='bold', color='#102a43', pad=14)

    # Outer spines
    for spine in ax.spines.values():
        spine.set_color('#334e68')
        spine.set_linewidth(1.3)

    plt.tight_layout()
    out_path = 'docs/images/amc_top_10_features_mockup_style.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    brain_dir = r"C:\Users\Dell\.gemini\antigravity-ide\brain\a8dd8184-259a-4c26-94e7-f02adf5165bb"
    shutil.copy(out_path, os.path.join(brain_dir, "amc_top_10_features_mockup_style.png"))
    print(f"Saved compact version: {out_path}")

if __name__ == '__main__':
    plot_top10_features_compact()
