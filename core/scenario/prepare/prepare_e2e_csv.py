import pandas as pd
import yaml
import os

def prepare_e2e_csv(args):
    path = os.path.join(args.experiment_dir, f'{args.experiment}.yaml')
    with open(path, 'r') as fp:
        experiment_yaml = yaml.safe_load(fp)
    LBS = experiment_yaml['LBS']
    UBS = experiment_yaml['UBS']
    PNRS = experiment_yaml['PNRS']
    N_LATENTS = experiment_yaml['N_LATENTS']
    METHODS = ['e2e']
    data = {
        'onnx': [],
        'vnnlib': [],
        'timeout': [],
        'adv_snr_power': [],
        'n_latent': [],
        'lb': [],
        'ub': [],
    }
    for adv_snr_power in PNRS:
        for lb, ub in zip(LBS, UBS):
            for n_latent in N_LATENTS:
                for i in range(10):
                    for j in range(args.n_sample_per_class):
                        for method in METHODS:
                            adv_snr_power = float(adv_snr_power)
                            lb = float(lb)
                            ub = float(ub)
                            n_latent = int(n_latent)
                            onnx = os.path.join(args.onnx_dir.replace(args.base_folder, '..'),
                                                f'{args.experiment}_{method}_{adv_snr_power:0.1f}_{n_latent}_{args.n_channel}_{lb}_{ub}',
                                    f'net.onnx')
                            vnnlib = os.path.join(args.vnnlib_dir.replace(args.base_folder, '..'),
                                                  f'{args.experiment}_{method}_{adv_snr_power:0.1f}_{n_latent}_{args.n_channel}_{lb}_{ub}',
                                    f'{i}_{j}.vnnlib')
                            if os.path.exists(onnx) and os.path.exists(vnnlib):
                                timeout = args.timeout
                                data['onnx'].append(onnx)
                                data['vnnlib'].append(vnnlib)
                                data['timeout'].append(timeout)
                                data['adv_snr_power'].append(adv_snr_power)
                                data['n_latent'].append(n_latent)
                                data['lb'].append(lb)
                                data['ub'].append(ub)
                            else:
                                print(f'[!] missing {onnx}')
    df = pd.DataFrame(data)
    path = os.path.join(f'{args.vnnlib_dir}', f'{args.experiment}_{method}.csv')
    df.to_csv(path, index=None)
