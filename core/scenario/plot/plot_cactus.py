import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from . import util

def plot_cactus(args):
    dfs, stats = util.get_dfs(args)
    plt.figure(figsize=util.FIG_SIZE)
    max_solve = 0
    for v in util.VERIFIER_NAMES:
        sat = dfs[v][dfs[v]['sat'] == 1]
        unsat = dfs[v][dfs[v]['unsat'] == 1]
        stats[v]['verified'] += len(unsat)
        stats[v]['falsified'] += len(sat)
        solved = pd.concat([sat, unsat])
        values = np.clip(np.sort(solved['dt'].values), max=util.TIMEOUTS[v])
        plt.plot(values, label=util.VERIFIER_NAMES[v], linewidth=1.2)
        max_solve = max(max_solve, len(solved))
    plt.xlim(0, int(max_solve * 1.1))
    plt.ylabel("Runtime (s)", fontsize=util.FONT_SIZE)
    plt.xlabel("Solved problems", fontsize=util.FONT_SIZE)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.xticks(fontsize=util.FONT_SIZE)
    plt.tight_layout()
    path = os.path.join(util.FIGURE_DIR, f'{args.scenario}_validated.pdf')
    plt.savefig(path, format='pdf')
    for v in util.VERIFIER_NAMES:
        print(f'{util.VERIFIER_NAMES[v]} & {stats[v]["verified"]} & {stats[v]["falsified"]} & {stats[v]["incorrect"]} \\\\')
    print(f'[plot_cactus] saved {path=}')
    plt.close()
    legend = plt.figure(figsize=(util.FIG_SIZE[0], 0.17))
    legend.legend(handles, labels, loc='center', ncol=4, frameon=False, fontsize=util.FONT_SIZE,
                  handlelength=1.5, columnspacing=1.5, borderpad=0)
    legend.savefig(os.path.join(util.FIGURE_DIR, 'plot_cactus_legend.pdf'), format='pdf')
    plt.close(legend)
