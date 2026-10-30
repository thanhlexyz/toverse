import pandas as pd
import simulator
import time
import os

def verify_to_export_cex(args):
    if args.verifier in ['e2e_mip']:
        path = os.path.join(f'{args.vnnlib_dir}', f'{args.experiment}_e2e.csv')
    else:
        raise NotImplementedError
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
                if info['sat'] == 1:
                    simulator.verifier.verify(onnx, vnnlib, timeout, args)
                total_sat += info['sat']
                total_unsat += info['unsat']
                total_timeout += info['timeout']
                info['total_sat'] = total_sat
                info['total_unsat'] = total_unsat
                info['total_timeout'] = total_timeout
                monitor.step(info)
        except KeyboardInterrupt:
            exit()
