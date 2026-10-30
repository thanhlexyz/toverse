import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


from . import util

def plot_boxplot(args):
    dfs_orig, _ = util.get_dfs(args)
    data = []
    tick_labels = []
    
    mask = (dfs_orig['e2e']['unsat'] == 1) & (dfs_orig['e2e_mip']['unsat'] == 1) & (dfs_orig['abcrown']['unsat'] == 1)
    
    new_dfs = {}
    new_dfs['abcrown'] = dfs_orig['abcrown'].loc[mask].reset_index(drop=True)
    new_dfs['e2e'] = dfs_orig['e2e'].loc[mask].reset_index(drop=True)
    new_dfs['e2e_mip'] = dfs_orig['e2e_mip'].loc[mask].reset_index(drop=True)
    
    
    for v in new_dfs:
        unsat = new_dfs[v][new_dfs[v]['unsat'] == 1]
        solved = unsat
        solved_dt = (solved['dt'].values / util.TIMEOUTS[v]).tolist()
        timeout = new_dfs[v][new_dfs[v]['timeout'] == 1]
        timeout_dt = (np.clip(timeout['dt'].values, a_min=None, a_max=util.TIMEOUTS[v]) / util.TIMEOUTS[v]).tolist()
        data.append(solved_dt + timeout_dt)
        print(f'{v=} {len(data[-1])=}')
        tick_labels.append(util.VERIFIER_NAMES[v])
    plt.figure(figsize=(5, 3))
    plt.boxplot(data, labels=tick_labels, showfliers=False)
    plt.ylabel("Runtime (s)")
    plt.xlabel("Verifier")
    plt.tight_layout()
    path = os.path.join(util.FIGURE_DIR, f'{args.scenario}_validated.pdf')
    plt.savefig(path, format='pdf')
    print(f'[plot_boxplot] saved {path=}')
    plt.close()
