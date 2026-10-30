import pandas as pd
import numpy as np

def check_invalid_split(args):
    df_e2e_mip = pd.read_csv('../data/csv/e2e_mip_fashion_mnist.csv')
    df_split = pd.read_csv('../data/csv/split_fashion_mnist.csv')

    split_unsat  = df_split['unsat'].to_numpy()
    e2e_mip_sat = df_e2e_mip['sat'].to_numpy()
    mask = np.logical_and(split_unsat, e2e_mip_sat)
    print(mask.sum())
