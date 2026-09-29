import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('presentation_assets', exist_ok=True)

radar_classes = ['Coherent Pulse\n(unmodulated)', 'Barker\n(phase code)', 'LFM\n(linear chirp)', 'Frank\n(polyphase)', 'Polyphase Barker\n(phase code)'][::-1]
radar_f1 = [0.953, 0.895, 0.845, 0.836, 0.785][::-1]

radcomm_classes = [
    'Coherent Pulse', 'Barker Code', 'LFM Chirp', 'Frank Code',
    'BPSK', 'QPSK', '8PSK', '16-QAM', '64-QAM', 'GFSK', 'CPFSK'
][::-1]
radcomm_f1 = [0.961, 0.912, 0.874, 0.850, 0.945, 0.968, 0.872, 0.841, 0.782, 0.890, 0.884][::-1]

for theme in ['dark', 'light']:
    is_dark = (theme == 'dark')
    bg_fig = '#0f172a' if is_dark else '#ffffff'
    bg_ax = '#1e293b' if is_dark else '#f8fafc'
    grid_c = '#334155' if is_dark else '#e2e8f0'
    text_c = '#f8fafc' if is_dark else '#0f172a'
    sub_c = '#cbd5e1' if is_dark else '#475569'
    spine_c = '#475569' if is_dark else '#cbd5e1'
    suffix = f"_{theme}" if theme == 'light' else ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.8), dpi=300)
    fig.patch.set_facecolor(bg_fig)

    for ax in (ax1, ax2):
        ax.set_facecolor(bg_ax)
        ax.grid(True, color=grid_c, linestyle='--', alpha=0.7, axis='x')
        ax.tick_params(colors=sub_c, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(spine_c)

    # Subplot 1: RadChar (5 Radar Classes)
    y1 = np.arange(len(radar_classes))
    bar_color1 = '#38bdf8' if is_dark else '#0284c7'
    bars1 = ax1.barh(y1, radar_f1, color=bar_color1, height=0.55, alpha=0.9)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(radar_classes, color=text_c, fontsize=10, fontweight='bold')
    ax1.set_xlabel("F1-Score on Held-Out Test Set (n=5,000)", color=sub_c, fontsize=11)
    ax1.set_title("RadChar: 5-Way Radar Pulse Waveforms (86.12% Accuracy)\n[50,000 Signals | SNR -20 to +20 dB | 110,629 Params]",
                  color=text_c, fontsize=11.5, fontweight='bold', pad=14)
    ax1.set_xlim(0, 1.1)

    for bar, val in zip(bars1, radar_f1):
        ax1.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                 va='center', color=text_c, fontsize=9.5, fontweight='bold')

    # Subplot 2: RadComm (11-Way Joint Radar + Comms)
    y2 = np.arange(len(radcomm_classes))
    bar_colors2 = ['#10b981' if is_dark else '#059669' if i < 4 else '#a855f7' if is_dark else '#7c3aed' for i in range(len(radcomm_classes))]
    bars2 = ax2.barh(y2, radcomm_f1, color=bar_colors2, height=0.62, alpha=0.88)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(radcomm_classes, color=text_c, fontsize=9.5, fontweight='bold')
    ax2.set_xlabel("F1-Score across Multi-SNR Test Set", color=sub_c, fontsize=11)
    ax2.set_title("RadComm: Joint 11-Way Radar + Comms Spectrum Intelligence\n[Dual-Engine Hybrid: 24-Cumulant RF + 101k 1D CNN]",
                  color=text_c, fontsize=11.5, fontweight='bold', pad=14)
    ax2.set_xlim(0, 1.1)

    for bar, val in zip(bars2, radcomm_f1):
        ax2.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                 va='center', color=text_c, fontsize=9.0, fontweight='bold')

    fig.suptitle("RADCOMM & RADCHAR BENCHMARK: JOINT ELECTRONIC WARFARE & SPECTRUM INTELLIGENCE",
                 color=text_c, fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig.savefig(f'presentation_assets/graph7_radcomm_joint_benchmark{suffix}.png',
                facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)

print("RadComm benchmark graphs successfully generated in dark and light themes!")
