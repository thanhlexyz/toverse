import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from . import util

def plot_ablation(args):
    path = os.path.join(args.csv_dir, 'e2e_mip_ablation_fashion_mnist.csv')
    df = pd.read_csv(path)

    df = df[(df['pnr'] >= -10.0) & (df['pnr'] <= -2.5)]
    grouped = df.groupby('pnr')[['sat', 'unsat', 'timeout']].sum().reset_index()
    grouped = grouped.sort_values('pnr')

    import numpy as np
    x = np.arange(len(grouped))
    width = 0.25

    fig, ax = plt.subplots(figsize=util.FIG_SIZE)
    ax.bar(x - width, grouped['unsat'],   width=width, color=sns.color_palette()[0], label='Verified')
    ax.bar(x,         grouped['sat'],     width=width, color=sns.color_palette()[1],   label='Falsified')
    ax.bar(x + width, grouped['timeout'], width=width, color=sns.color_palette()[2], label='Timeout')

    ax.set_xticks(x)
    ax.set_xticklabels(grouped['pnr'].tolist(), fontsize=util.FONT_SIZE)
    ax.set_xlabel('PNR (dB)', fontsize=util.FONT_SIZE)
    ax.set_ylabel('Number of problems', fontsize=util.FONT_SIZE)
    ax.legend(fontsize=util.FONT_SIZE, loc='upper center', bbox_to_anchor=(0.5, 1.26), frameon=False, ncol=3)
    plt.xticks(fontsize=util.FONT_SIZE)
    plt.yticks(fontsize=util.FONT_SIZE)
    ax.set_yticks([0, 20, 40, 60, 80])
    plt.tight_layout()

    out = os.path.join(util.FIGURE_DIR, f'{args.scenario}.pdf')
    plt.savefig(out, format='pdf')
    print(f'[plot_ablation] saved to {out}')
    plt.close()
