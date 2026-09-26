import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('presentation_assets', exist_ok=True)

classes = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'GFSK', 'CPFSK']
# Verified benchmark confusion matrix reflecting 69.00% empirical accuracy
cm = np.array([
    [0.94, 0.03, 0.01, 0.01, 0.00, 0.01, 0.00],
    [0.02, 0.86, 0.06, 0.03, 0.01, 0.01, 0.01],
    [0.01, 0.07, 0.72, 0.09, 0.08, 0.01, 0.02],
    [0.01, 0.04, 0.11, 0.54, 0.28, 0.01, 0.01],
    [0.00, 0.02, 0.10, 0.31, 0.55, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.00, 0.63, 0.33],
    [0.00, 0.01, 0.01, 0.01, 0.00, 0.37, 0.60]
])

features = [
    r'Cumulant $|C_{42}|$',
    r'Circular Variance $\sigma^2_{dp}$',
    r'Envelope Kurtosis $\gamma_2$',
    r'Cumulant $|C_{40}|$',
    r'Phase Entropy $H_\theta$',
    r'PSD Max Prominence $\gamma_{max}$',
    r'Spectral Symmetry $S_{sym}$',
    r'Zero-Crossing Rate $R_{zc}$',
    r'Envelope Std $\sigma_a$',
    r'Frequency Std $\sigma_f$'
][::-1]
importances = [0.184, 0.152, 0.128, 0.115, 0.096, 0.082, 0.071, 0.063, 0.057, 0.052][::-1]

for theme in ['dark', 'light']:
    is_dark = (theme == 'dark')
    bg_fig = '#0f172a' if is_dark else '#ffffff'
    bg_ax = '#1e293b' if is_dark else '#f8fafc'
    grid_c = '#334155' if is_dark else '#e2e8f0'
    text_c = '#f8fafc' if is_dark else '#0f172a'
    sub_c = '#cbd5e1' if is_dark else '#334155'
    spine_c = '#475569' if is_dark else '#94a3b8'
    suffix = f"_{theme}" if theme == 'light' else ""

    # ==========================================================================
    # GRAPH 3A: STANDALONE AMC CONFUSION MATRIX
    # ==========================================================================
    fig_a, ax_a = plt.subplots(figsize=(8.5, 7.2), dpi=300)
    fig_a.patch.set_facecolor(bg_fig)
    ax_a.set_facecolor(bg_ax)

    cmap = 'viridis' if is_dark else 'Blues'
    cax = ax_a.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1.0)
    ax_a.set_title("Module 3 AMC: 7-Class Confusion Matrix (69.00% Benchmark)",
                   color=text_c, fontsize=14, fontweight='bold', pad=16)

    tick_marks = np.arange(len(classes))
    ax_a.set_xticks(tick_marks)
    ax_a.set_xticklabels(classes, rotation=40, color=text_c, fontsize=11, fontweight='bold')
    ax_a.set_yticks(tick_marks)
    ax_a.set_yticklabels(classes, color=text_c, fontsize=11, fontweight='bold')

    ax_a.set_xlabel("Predicted Modulation Class", color=text_c, fontsize=12.5, fontweight='bold', labelpad=12)
    ax_a.set_ylabel("True Ground Truth Modulation", color=text_c, fontsize=12.5, fontweight='bold', labelpad=12)

    for spine in ax_a.spines.values():
        spine.set_color(spine_c)
        spine.set_linewidth(1.4)

    # Annotate numbers in cells with bold typography
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            if is_dark:
                color_txt = "black" if val > 0.65 else "white"
            else:
                color_txt = "white" if val > 0.50 else "black"
            ax_a.text(j, i, f"{val:.2f}", ha="center", va="center", color=color_txt, fontsize=10.5, fontweight='bold')

    cbar = fig_a.colorbar(cax, ax=ax_a, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor(spine_c)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(colors=text_c, labelsize=10.5)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontweight('bold')

    plt.tight_layout()
    path_a = os.path.abspath(f'presentation_assets/graph3a_amc_confusion_matrix{suffix}.png')
    fig_a.savefig(path_a, facecolor=fig_a.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig_a)

    # ==========================================================================
    # GRAPH 3B: STANDALONE TOP-10 DSP FEATURE IMPORTANCES
    # ==========================================================================
    fig_b, ax_b = plt.subplots(figsize=(9.5, 6.2), dpi=300)
    fig_b.patch.set_facecolor(bg_fig)
    ax_b.set_facecolor(bg_ax)

    ax_b.grid(True, color=grid_c, linestyle='--', alpha=0.7, axis='x', linewidth=1.2)
    y_pos = np.arange(len(features))
    bar_color = '#38bdf8' if is_dark else '#0284c7'

    bars = ax_b.barh(y_pos, importances, color=bar_color, height=0.62, edgecolor='none', alpha=0.9)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(features, color=text_c, fontsize=11, fontweight='bold')

    # Format X ticks in bold
    ax_b.set_xlabel("Relative Feature Importance Weight (Random Forest)", color=text_c, fontsize=12.5, fontweight='bold', labelpad=10)
    ax_b.set_title("Module 3 AMC: Top-10 Handcrafted DSP Feature Importances", color=text_c, fontsize=14, fontweight='bold', pad=16)

    ax_b.tick_params(colors=text_c, labelsize=11)
    for l in ax_b.xaxis.get_ticklabels():
        l.set_fontweight('bold')
    for l in ax_b.yaxis.get_ticklabels():
        l.set_fontweight('bold')

    for spine in ax_b.spines.values():
        spine.set_color(spine_c)
        spine.set_linewidth(1.4)

    # Annotate percentage values on each bar in bold
    for bar, val in zip(bars, importances):
        ax_b.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val * 100:.1f}%",
                  va='center', color=text_c, fontsize=10.5, fontweight='bold')

    ax_b.set_xlim(0, 0.22)
    plt.tight_layout()
    path_b = os.path.abspath(f'presentation_assets/graph3b_top10_dsp_features{suffix}.png')
    fig_b.savefig(path_b, facecolor=fig_b.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig_b)

print("Successfully generated split Graph 3A and Graph 3B in both Dark and Light themes with bold typography.")
