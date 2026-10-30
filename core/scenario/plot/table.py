import pandas as pd
import numpy as np
import os

def table(args):
    verifiers = ['e2e', 'abcrown', 'e2e_mip']
    labels = {'e2e': 'E2E', 'abcrown': '$\\alpha\\beta-CROWN$', 'e2e_mip': 'E2E-MIP'}
    dfs = {}
    for v in verifiers:
        path = os.path.join(args.csv_dir, f'{v}_{args.experiment}.csv')
        dfs[v] = pd.read_csv(path)
    for v in verifiers:
        sat = dfs[v][dfs[v]['sat'] == 1]
        unsat = dfs[v][dfs[v]['unsat'] == 1]
        timeout = dfs[v][dfs[v]['timeout'] == 1]
        if args.experiment == 'fashion_mnist':
            if v == 'e2e_mip':
                invalid = []
            else:
                invalid = pd.concat([sat])
        else:
            invalid = []

        line = f'{labels[v]} & {len(unsat)} & {len(sat)} & {len(timeout)} & {len(invalid)} \\\\ '
        print(line)
    print('\\bottomrule')
