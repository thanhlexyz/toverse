import seaborn as sns
import pandas as pd
import matplotlib
import os

sns.set_theme(style="whitegrid")
sns.set_palette("muted")
FONT_SIZE = 8

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times']
matplotlib.rcParams['font.size'] = FONT_SIZE
matplotlib.rcParams['axes.labelsize'] = FONT_SIZE
matplotlib.rcParams['xtick.labelsize'] = FONT_SIZE
matplotlib.rcParams['ytick.labelsize'] = FONT_SIZE
matplotlib.rcParams['legend.fontsize'] = FONT_SIZE
matplotlib.rcParams['axes.linewidth'] = 0.6
matplotlib.rcParams['grid.linewidth'] = 0.4


TIMEOUTS = {
    "pgm": 6.0,
    "abcrown": 60.0,
    "e2e": 60.0,
    "e2e_mip": 60.0,
}

FIG_SIZE = (3.5, 1.7)
SUBFIG_SIZE = (1.71, 1.45)

VERIFIER_NAMES = {
    "pgm": r"\textsc{Pgm}",
    "abcrown": r"\textsc{$\alpha\beta$-Crown}",
    "e2e": r"\textsc{NeuralSAT}",
    "e2e_mip": r"\textsc{Toverse}",
}

FIGURE_DIR = '../../paper/figure/result/'

def get_dfs(args):
    experiments = ['fashion_mnist', 'cifar10']
    dfs = {v: [] for v in VERIFIER_NAMES}
    stats = {v: {'verified': 0, 'falsified': 0, 'incorrect': 0} for v in VERIFIER_NAMES}
    for v in VERIFIER_NAMES:
        for experiment in experiments:
            path = os.path.join(args.csv_dir, f'{v}_{experiment}.csv')
            df_ = pd.read_csv(path)
            if v in ['abcrown', 'e2e'] and experiment == 'fashion_mnist':
                indices = df_[df_['sat'] == 1].index
                df_.loc[indices, 'sat'] = 0
                df_.loc[indices, 'timeout'] = 1
                stats[v]['incorrect'] += len(indices)

            over_time_mask = df_['dt'] >= TIMEOUTS[v]
            df_.loc[over_time_mask, 'sat'] = 0
            df_.loc[over_time_mask, 'unsat'] = 0
            df_.loc[over_time_mask, 'timeout'] = 1

            dfs[v].append(df_)
        dfs[v] = pd.concat(dfs[v])
    return dfs, stats
