import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import re
import os

from . import util

def _log_path(vnnlib_path: str, verifier: str) -> str:
    return vnnlib_path.replace('/vnnlib/', f'/log/{verifier}/').replace('.vnnlib', '.log')


def _count_iterations(log_path: str) -> int:
    text = open(log_path).read()
    hits = re.findall(r'Iteration:\s+(\d+)', text)
    return int(hits[-1]) if hits else 0


def _plot_legend():
    name = util.VERIFIER_NAMES['e2e_mip']
    handles = [Patch(color='green', alpha=0.4, label=f'{name} fewer'),
               Patch(color='red', alpha=0.4, label=f'{name} more')]
    fig = plt.figure(figsize=(util.FIG_SIZE[0], 0.17))
    fig.legend(handles=handles, loc='center', ncol=2, frameon=False, fontsize=util.FONT_SIZE,
               handlelength=1.0, columnspacing=1.5, borderpad=0)
    fig.savefig(os.path.join(util.FIGURE_DIR, 'plot_compare_legend.pdf'), format='pdf')
    plt.close(fig)


def plot_compare_iteration(args):
    dfs, _ = util.get_dfs(args)
    
    mask = (dfs['e2e']['unsat'] == 1) & (dfs['e2e_mip']['unsat'] == 1)
    e2e_rows     = dfs['e2e'].loc[mask].reset_index(drop=True)
    e2e_mip_rows = dfs['e2e_mip'].loc[mask].reset_index(drop=True)

    e2e_iters     = np.array([_count_iterations(_log_path(r['vnnlib'], 'e2e'))
                               for _, r in e2e_rows.iterrows()])
    e2e_mip_iters = np.array([_count_iterations(_log_path(r['vnnlib'], 'e2e_mip'))
                               for _, r in e2e_mip_rows.iterrows()])
    
    delta = e2e_mip_iters - e2e_iters
    delta = delta[delta != 0]
    order = np.argsort(delta)
    delta = delta[order]
    x = np.arange(len(delta))

    fig, ax = plt.subplots(figsize=util.SUBFIG_SIZE)
    ax.plot(x, delta, color='black', linewidth=0.2)
    if any(delta >= 0):
        ax.fill_between(x, delta, 0, where=(delta >= 0), color='red',   alpha=0.4, label=f'{util.VERIFIER_NAMES["e2e_mip"]} more')
    if any(delta <  0):
        ax.fill_between(x, delta, 0, where=(delta <  0), color='green', alpha=0.4, label=f'{util.VERIFIER_NAMES["e2e_mip"]} fewer')
    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xlabel('Verified problems', fontsize=util.FONT_SIZE)
    ax.set_ylabel('$\\Delta$ iterations', fontsize=util.FONT_SIZE)
    plt.xlim(-5, int(len(delta))+10)
    plt.ylim(int(min(delta) * 1.2), int(max(delta) * 1.6))
    plt.xticks(fontsize=util.FONT_SIZE)
    plt.yticks(fontsize=util.FONT_SIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    plt.tight_layout()
    out = os.path.join(util.FIGURE_DIR, f'{args.scenario}.pdf')
    plt.savefig(out, format='pdf')
    print(f'[plot_compare_iteration] saved to {out}')
    plt.close()
    _plot_legend()
