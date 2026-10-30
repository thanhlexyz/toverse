
import pandas as pd
import numpy as np
import torch
import re
import os

def parse_z_from_text(text, n_latent):
    matches = re.findall(r'\(X_(\d+)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)', text)
    matches.sort(key=lambda m: int(m[0]))
    inputs = np.array([float(v) for _, v in matches], dtype=np.float32)
    z = inputs[-n_latent:]
    return torch.from_numpy(z)

def noise_constraint_fn(z, constraint_data):
    max_power = 10 ** (constraint_data['adv_snr_threshold'] / 10) * constraint_data['z2_ub']
    if (z ** 2).sum().item() > max_power:
        return 0
    if (z > torch.from_numpy(constraint_data['z_ub']).to(z.device)).any():
        return 0
    if (z < torch.from_numpy(constraint_data['z_lb']).to(z.device)).any():
        return 0
    return 1

def check_cex(args):
    path = os.path.join(args.csv_dir, f'{args.verifier}_{args.experiment}.csv')
    df = pd.read_csv(path)
    for i, row in df.iterrows():
        path = os.path.join(os.path.dirname(row['onnx'].replace('e2e', 'split')), 'z_constraint.pt')
        constraint_data = torch.load(path, map_location='cpu', weights_only=False)
        break
    valids = []
    for i, row in df.iterrows():
        if row['sat'] == 1:
            if args.experiment == 'fashion_mnist':
                path = row['onnx'].replace('/onnx/', f'/log/{args.verifier}/').replace('.onnx', '.res')
            elif args.experiment == 'cifar10':
                name = os.path.basename(row['vnnlib']).replace('.vnnlib', '.res')
                folder = os.path.dirname(row['onnx']).replace('/onnx/', f'/log/{args.verifier}/')
                path = os.path.join(folder, name)
            else:
                raise NotImplementedError
            n_latent = row['n_latent']
            text = open(path).read().strip()
            try:
                z = parse_z_from_text(text, n_latent)
                valid = noise_constraint_fn(z, constraint_data)
                print(f'{i},{valid}')
            except:
                valid = 1
            valids.append(valid)
        else:
            valids.append(1)
    df['valid'] = valids
    path = os.path.join(args.csv_dir, f'{args.verifier}_{args.experiment}_validated.csv')
    df.to_csv(path, index=None)
