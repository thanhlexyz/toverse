import os

from . import util

def exp_prepare(args):
    for experiment in [args.experiment]:
        print(f'\n[+] prepare verification properties for {experiment}')
        args.experiment = experiment
        config = util.load_exp_yaml(args)
        linf_eps = config['LINF_EPS']
        for adv_snr_threshold in config['PNRS']:
            for r_lb, r_ub in zip(config['LBS'], config['UBS']):
                for n_latent in config['N_LATENTS']:
                    cmd =  f'python3 -W ignore main.py '
                    cmd += f'--scenario=prepare_e2e_benchmark '
                    cmd += f'--dataset={config["DATASET"]} '
                    cmd += f'--{adv_snr_threshold=} '
                    cmd += f'--{linf_eps=} '
                    cmd += f'--{r_lb=} '
                    cmd += f'--{r_ub=} '
                    cmd += f'--{n_latent=} '
                    cmd += f'--experiment={args.experiment} '
                    cmd += f'--n_sample_per_class={args.n_sample_per_class} '
                    cmd += f'--no_pbar '
                    print(cmd)
                    os.system(cmd)
        for adv_snr_threshold in config['PNRS']:
            for r_lb, r_ub in zip(config['LBS'], config['UBS']):
                for n_latent in config['N_LATENTS']:
                    cmd =  f'python3 -W ignore main.py '
                    cmd += f'--scenario=prepare_split_benchmark '
                    cmd += f'--dataset={config["DATASET"]} '
                    cmd += f'--{adv_snr_threshold=} '
                    cmd += f'--{linf_eps=} '
                    cmd += f'--{r_lb=} '
                    cmd += f'--{r_ub=} '
                    cmd += f'--{n_latent=} '
                    cmd += f'--experiment={args.experiment} '
                    cmd += f'--n_sample_per_class={args.n_sample_per_class} '
                    cmd += f'--no_pbar '
                    print(cmd)
                    os.system(cmd)

        for method in ['e2e', 'split']:
            cmd  = f'python3 -W ignore main.py '
            cmd += f'--scenario=prepare_{method}_csv '
            cmd += f'--experiment={args.experiment} '
            cmd += f'--n_sample_per_class={args.n_sample_per_class} '
            print(cmd)
            os.system(cmd)
