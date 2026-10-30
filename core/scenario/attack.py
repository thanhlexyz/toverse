from simulator.util.spec.objective import parse_vnnlib
from simulator.util.read_onnx import parse_onnx
import simulator

import pandas as pd
import torch
import tqdm
import time
import os


def attack(args):
    path = os.path.join(f'{args.vnnlib_dir}', f'{args.experiment}_e2e.csv')
    print(f'[ATTACK] {path=}')
    df = pd.read_csv(path)
    monitor = simulator.MonitorV(len(df), args)
    solved_df = monitor.load_csv()
    if solved_df is not None:
        solved_instances = solved_df['instance'].tolist()
    else:
        solved_instances = []
    total_sat, total_unsat, total_timeout = 0, 0, 0
    for i, (onnx, vnnlib, timeout, pnr, n_latent, lb, ub) in df.iterrows():
        try:
            if i in solved_instances:
                info = {'instance': i,
                        'onnx': onnx,
                        'vnnlib': vnnlib,
                        'pnr': pnr,
                        'n_latent': n_latent,
                        'lb': lb,
                        'ub': ub,
                        'dt': solved_df.loc[i, 'dt'],
                        'sat': solved_df.loc[i, 'sat'],
                        'unsat': solved_df.loc[i, 'unsat'],
                        'timeout': solved_df.loc[i, 'timeout'],
                }
            else:
                try:
                    net, input_shape, output_shape = parse_onnx(onnx)
                    net = net.to(args.device)
                    objectives = parse_vnnlib(vnnlib, input_shape)
                    args.adv_snr_threshold = float(pnr)
                    args.n_latent          = n_latent
                    args.r_lb              = lb
                    args.r_ub              = ub
                    tic = time.time()
                    is_attacked, _ = simulator.verifier.attack(net, objectives, input_shape, timeout, args)
                    toc = time.time()
                except AssertionError:
                    tic = toc = time.time()
                    is_attacked = False
                    print('[!] assertion error:', onnx)
                except:
                    raise NotImplementedError
                info = {'instance': i,
                        'onnx': onnx,
                        'vnnlib': vnnlib,
                        'pnr': pnr,
                        'n_latent': n_latent,
                        'lb': lb,
                        'ub': ub,
                        'dt': toc - tic,
                        'sat': int(is_attacked),
                        'unsat': 0,
                        'timeout': int(not is_attacked),
                }
            total_sat += info['sat']
            total_unsat += info['unsat']
            total_timeout += info['timeout']
            info['total_sat'] = total_sat
            info['total_unsat'] = total_unsat
            info['total_timeout'] = total_timeout
            monitor.step(info)
            monitor.export_csv()
        except KeyboardInterrupt:
            exit()
