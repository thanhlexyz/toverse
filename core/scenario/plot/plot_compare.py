import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

from . import util

def _log_path(vnnlib_path: str, verifier: str) -> str:
    return vnnlib_path.replace('/vnnlib/', f'/log/{verifier}/').replace('.vnnlib', '.log')


def _count_runtime(log_path: str) -> int:
    text = open(log_path).read()
    hits = re.findall(r'Runtime:\s+(\d+\.\d+)', text)
    return float(hits[-1])

def plot_compare(args):
    dfs, _ = util.get_dfs(args)

    mask = (dfs['e2e']['unsat'] == 1) & (dfs['e2e_mip']['unsat'] == 1)
    e2e_rows     = dfs['e2e'].loc[mask].reset_index(drop=True)
    e2e_mip_rows = dfs['e2e_mip'].loc[mask].reset_index(drop=True)

    e2e_runtime     = np.array([_count_runtime(_log_path(r['vnnlib'], 'e2e'))
                                  for _, r in e2e_rows.iterrows()])
    e2e_mip_runtime = np.array([_count_runtime(_log_path(r['vnnlib'], 'e2e_mip'))
                                  for _, r in e2e_mip_rows.iterrows()])

    delta = e2e_mip_runtime - e2e_runtime
    threshold = 0.5
    delta = delta[(delta > 0) | (delta < -threshold)]
    order = np.argsort(delta)
    delta = delta[order]
    x = np.arange(len(delta))
    
    fig, ax = plt.subplots(figsize=util.FIG_SIZE)
    ax.plot(x, delta, color='black', linewidth=0.5)
    if any(delta >= 0):
        ax.fill_between(x, delta, 0, where=(delta >= 0), color='red',   alpha=0.5, label=f'{util.VERIFIER_NAMES["e2e_mip"]} slower')
    if any(delta <  0):
        ax.fill_between(x, delta, 0, where=(delta <  0), color='green', alpha=0.5, label=f'{util.VERIFIER_NAMES["e2e_mip"]} faster')
    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xlabel('Verified instances', fontsize=util.FONT_SIZE)
    ax.set_ylabel('$\\Delta$ runtime (s)', fontsize=util.FONT_SIZE)
    ax.legend(fontsize=util.FONT_SIZE)
    plt.xlim(-10, int(len(delta) +10))
    plt.ylim(int(min(delta) * 1.1), int(max(delta) * 1.5))
    plt.xticks(fontsize=util.FONT_SIZE)
    plt.yticks(fontsize=util.FONT_SIZE)
    plt.tight_layout()
    path = os.path.join(util.FIGURE_DIR, f'{args.scenario}.pdf')
    plt.savefig(path, format='pdf')
    print(f'[plot_compare] saved to {path}')
