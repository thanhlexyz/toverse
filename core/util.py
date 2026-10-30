import numpy as np
import argparse
import random
import torch
import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_printoptions(4, sci_mode=False)

def create_folders(args):
    ls = [args.data_dir, args.checkpoint_dir, args.onnx_dir,
          args.vnnlib_dir, args.csv_dir, args.figure_dir, args.log_dir]
    for folder in ls:
        if not os.path.exists(folder):
            os.makedirs(folder)

def set_random_seed(args):
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)

base_folder = os.path.dirname(os.path.dirname(__file__))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='train')
    parser.add_argument('--experiment', type=str, default='fashion_mnist', choices=['fashion_mnist', 'cifar10', 'ablation_fashion_mnist', 'ablation_cifar10'])
    parser.add_argument('--model', type=str, default='fashion_mnist_semcom')
    parser.add_argument('--trainer', type=str, default='semcom')
    parser.add_argument('--dataset', default='fashion_mnist', choices=['torch/cifar10', 'fashion_mnist'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--recon_w', type=float, default=0.1)
    parser.add_argument('--cutmix_sz', type=int, default=3)
    parser.add_argument('--cutmix_ep', type=int, default=6)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--n_channel', type=int, default=4)
    parser.add_argument('--base_channel_width', type=int, default=24)
    parser.add_argument('--n_latent', type=int, default=16)
    parser.add_argument('--attacker', type=str, default='pgm')
    parser.add_argument('--adv_snr_threshold', type=float, default=-10.0)
    parser.add_argument('--n_pgm_sample', type=float, default=100)
    parser.add_argument('--verifier', type=str, default='neuralsat')
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--path_loss', type=float, default=50.0)
    parser.add_argument('--n_vnnlib', type=int, default=100)
    parser.add_argument('--r_lb', type=float, default=-1.0)
    parser.add_argument('--r_ub', type=float, default=1.0)
    parser.add_argument('--linf_eps', type=float, default=0.01)
    parser.add_argument('--n_sample_per_class', type=int, default=100)

    parser.add_argument('--experiment_dir', type=str, default=f'{ROOT_DIR}/core/experiment')
    parser.add_argument('--checkpoint_dir', type=str, default=f'{ROOT_DIR}/data/checkpoint')
    parser.add_argument('--figure_dir', type=str, default=f'{ROOT_DIR}/data/figure')
    parser.add_argument('--vnnlib_dir', type=str, default=f'{ROOT_DIR}/data/vnnlib')
    parser.add_argument('--data_dir', type=str, default=f'{ROOT_DIR}/data/data')
    parser.add_argument('--onnx_dir', type=str, default=f'{ROOT_DIR}/data/onnx')
    parser.add_argument('--core_dir', type=str, default=f'{ROOT_DIR}/core')
    parser.add_argument('--csv_dir', type=str, default=f'{ROOT_DIR}/data/csv')
    parser.add_argument('--log_dir', type=str, default=f'{ROOT_DIR}/data/log')
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--no_pbar', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    create_folders(args)
    set_random_seed(args)
    args.base_folder = base_folder
    if args.dataset == 'fashion_mnist':
        args.input_shape = (1, 1, 28, 28)
    else:
        args.input_shape = (1, 3, 32, 32)

    if args.scenario == 'attack':
        args.trainer = args.attacker
    elif args.scenario == 'verify':
        args.trainer = args.verifier
    return args
