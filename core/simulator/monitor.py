import pandas as pd
import numpy as np
import torch
import tqdm
import os

def get_label(args):
    return f'train_{args.trainer}_{args.experiment}_{args.n_latent}_{args.adv_snr_threshold}'

class Monitor:

    def __init__(self, args):
        self.args = args
        if not args.no_pbar:
            T = args.max_epoch
            self.bar = tqdm.tqdm(range(T))
        self.csv_data = {}
        self.global_step = 0

    def __update_time(self):
        if not self.args.no_pbar:
            self.bar.update(1)

    def __update_description(self, **kwargs):
        _kwargs = {}
        for key in kwargs:
            if 'acc' in key:
                _kwargs[key] = f'{kwargs[key]:0.3f}'
            elif 'noise' in key:
                _kwargs[key] = f'{kwargs[key]:0.3f}'
            elif 'loss' in key:
                _kwargs[key] = f'{kwargs[key]:0.3f}'
            elif 'step' in key:
                _kwargs[key] = f'{kwargs[key]:d}'
        if not self.args.no_pbar:
            self.bar.set_postfix(**_kwargs)

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

    def __update_csv(self, info):
        for key in info.keys():
            if 'n_user_per_action' in key:
                pass
            else:
                if key not in self.csv_data:
                    self.csv_data[key] = [float(info[key])]
                else:
                    self.csv_data[key].append(float(info[key]))

    def export_csv(self):
        args = self.args
        path = os.path.join(args.csv_dir, f'{self.label}.csv')
        df = pd.DataFrame(self.csv_data)
        df.to_csv(path, index=None)
