import pandas as pd
import numpy as np
import torch
import tqdm
import os

def get_label(args):
    if args.scenario == 'attack':
        return f'{args.attacker}_{args.experiment}'
    elif 'verify' in args.scenario:
        return f'{args.verifier}_{args.experiment}'
    else:
        raise NotImplementedError

class MonitorV:

    def __init__(self, n_instance, args):
        self.args = args
        if not args.no_pbar:
            T = n_instance
            self.bar = tqdm.tqdm(range(T))
        self.csv_data = {}
        self.global_step = 0

    def __update_time(self):
        if not self.args.no_pbar:
            self.bar.update(1)

    def __display(self):
        if not self.args.no_pbar:
            self.bar.display()

    def step(self, info):
        self.__update_time()
        self.__update_description(**info)
        self.__display()
        self.__update_csv(info)
        self.global_step += 1

    @property
    def label(self):
        return get_label(self.args)

    def __update_description(self, **kwargs):
        _kwargs = {}
        for key in kwargs:
            if 'total_' in key:
                _kwargs[key] = f'{kwargs[key]:d}'
        if not self.args.no_pbar:
            self.bar.set_postfix(**_kwargs)

    def __update_csv(self, info):
        for key in info.keys():
            if 'total_' in key:
                continue
            if key not in self.csv_data:
                self.csv_data[key] = [info[key]]
            else:
                self.csv_data[key].append(info[key])

    def export_csv(self):
        args = self.args
        path = os.path.join(args.csv_dir, f'{self.label}.csv')
        df = pd.DataFrame(self.csv_data)
        df.to_csv(path, index=None)

    def load_csv(self):
        args = self.args
        path = os.path.join(args.csv_dir, f'{self.label}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df
