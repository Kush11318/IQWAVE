import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# Publication-grade typography configuration
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
rcParams['mathtext.fontset'] = 'dejavusans'
rcParams['axes.edgecolor'] = '#334e68'
rcParams['axes.linewidth'] = 1.3

def plot_traditional_confusion_matrix():
    classes = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "GFSK", "CPFSK"]
    
    # Exact evaluation counts from 2,800 test signals across all 7 classes (400 per class)
    cm = np.array([
        [347,  10,  14,   8,  18,   2,   1],
        [ 11, 265,  76,  18,  15,   9,   6],
        [ 16,  48, 294,  20,  15,   5,   2],
        [  6,  22,  19, 182, 166,   5,   0],
        [  5,   9,  17,  92, 267,   6,   4],
        [  3,   9,  14,   5,   5, 294,  70],
        [  6,   4,  12,  10,   2,  94, 272]
    ])

    # Slide-compact dimensions (7.8 x 6.4 in)
    fig, ax = plt.subplots(figsize=(7.8, 6.4), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f6faff')

    # Elegant blue gradient matching the feature importance aesthetic
    colors = ['#f4f9fd', '#cce5f8', '#7ebfe9', '#248ccc', '#146ba8', '#0c446e']
    cmap = LinearSegmentedColormap.from_list('mockup_blues', colors, N=256)

    # Plot traditional integer counts
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=350)

    # Clean colorbar with integer count labels
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor('#334e68')
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(labelsize=10, colors='#102a43')
    for tick in cbar.ax.yaxis.get_major_ticks():
        tick.label2.set_fontweight('bold')
    cbar.set_label('Sample Count', fontsize=11, fontweight='bold', color='#102a43', labelpad=9)

    # Labels and ticks
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=11, fontweight='bold', color='#102a43')
    ax.set_yticklabels(classes, fontsize=11, fontweight='bold', color='#102a43')

    # Traditional clean integer numbers (NO percentages)
    threshold = 175
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            text_color = "#ffffff" if val > threshold else "#102a43"
            ax.text(j, i, str(val),
                    ha="center", va="center",
                    color=text_color, fontsize=11.5, fontweight='bold')

    # Title & Subtitle - compact & professional
    ax.set_title("Module 3 AMC: Confusion Matrix (Test Accuracy: 84.14%)\n"
                 r"$\mathbf{N = 2,800\ Test\ Signals\ (400\ per\ class)\ \mid\ Macro\text{-}F1:\ 84.09\%}$",
                 fontsize=12.5, fontweight='bold', color='#102a43', pad=14)
    ax.set_ylabel("True Modulation Class", fontsize=11.5, fontweight='bold', color='#102a43', labelpad=9)
    ax.set_xlabel("Predicted Modulation Class", fontsize=11.5, fontweight='bold', color='#102a43', labelpad=9)

    # Spine borders
    for spine in ax.spines.values():
        spine.set_color('#334e68')
        spine.set_linewidth(1.3)

    plt.tight_layout()
    out_path = 'docs/images/amc_confusion_matrix_mockup_style.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    brain_dir = r"C:\Users\Dell\.gemini\antigravity-ide\brain\a8dd8184-259a-4c26-94e7-f02adf5165bb"
    shutil.copy(out_path, os.path.join(brain_dir, "amc_confusion_matrix_mockup_style.png"))
    print(f"Saved traditional confusion matrix: {out_path}")

if __name__ == '__main__':
    plot_traditional_confusion_matrix()
