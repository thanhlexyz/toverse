import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def get_status(row):
    if row['sat'] == 1: return 'sat'
    if row['unsat'] == 1: return 'unsat'
    if row['timeout'] == 1: return 'timeout'
    raise ValueError(f'Invalid status: {row}')

def compare_split(args):

    if 0:
        e2e_df = pd.read_csv(os.path.join(args.csv_dir.replace('/csv', '/exp_1/csv'), f'e2e_fashion_mnist.csv'))
        e2e_mip_df = pd.read_csv(os.path.join(args.csv_dir.replace('/csv', '/exp_1/csv'), f'e2e_mip_fashion_mnist.csv'))
    else:
        e2e_df = pd.read_csv(os.path.join(args.csv_dir, f'e2e_fashion_mnist.csv'))
        e2e_mip_df = pd.read_csv(os.path.join(args.csv_dir, f'e2e_mip_fashion_mnist.csv'))

    e2e_df['split'] = e2e_df.apply(get_status, axis=1)
    e2e_df = e2e_df.drop(columns=['sat', 'unsat', 'timeout']).rename(columns={'dt': 'dt_split'})

    e2e_mip_df['e2e_mip'] = e2e_mip_df.apply(get_status, axis=1)
    e2e_mip_df = e2e_mip_df.drop(columns=['sat', 'unsat', 'timeout']).rename(columns={'dt': 'dt_e2e_mip'})

    id_cols = ['instance', 'onnx', 'vnnlib', 'pnr', 'n_latent', 'lb', 'ub']
    merged_df = pd.merge(e2e_df, e2e_mip_df, on=id_cols)

    print('Total instances:', len(merged_df))
    merged_df.loc[(merged_df['split'] == 'sat'), 'e2e'] = 'timeout'

    timeout = 60.0
    merged_df.loc[(merged_df['dt_split'] > timeout), 'split'] = 'timeout'
    merged_df.loc[(merged_df['dt_e2e_mip'] > timeout), 'e2e_mip'] = 'timeout'

    mismatches = merged_df[(merged_df['e2e_mip'] == 'unsat') & (merged_df['split'] != 'unsat')]
    print('Mismatches:', len(mismatches))

    e2e_solved = merged_df[(merged_df['split'] != 'timeout')]
    print('Split solved:', len(e2e_solved))
    e2e_mip_solved = merged_df[(merged_df['e2e_mip'] != 'timeout')]
    print('E2E-MIP solved:', len(e2e_mip_solved))

    e2e_solved_dt = e2e_solved['dt_split'].values
    e2e_mip_solved_dt = e2e_mip_solved['dt_e2e_mip'].values

    plt.figure(figsize=(8, 6))
    plt.plot(np.sort(e2e_solved_dt), label='Split')
    plt.plot(np.sort(e2e_mip_solved_dt), label='E2E-MIP')
    plt.legend()
    plt.xlabel('solved instances')
    plt.ylabel('runtime (s)')
    plt.savefig(os.path.join(args.figure_dir, 'compare-split.png'))
    plt.close()
