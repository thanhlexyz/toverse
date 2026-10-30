import torch.nn as nn
import numpy as np
import simulator
import pickle
import torch
import tqdm
import os

from simulator.util.milp_solver import build_milp_solver
from simulator.model.perturbation import PerturbationLayer

class SemCom(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        args.trainer = 'pgm'
        trainer = simulator.trainer.create(args)
        trainer.load()
        self.encoder    = trainer.semcom.vae.encoder.cpu()
        self.decoder    = trainer.semcom.vae.decoder.cpu()
        self.classifier = trainer.semcom.pragmatic.cpu()
        self.generator  = trainer.generator.cpu()
        self.trainer = trainer

    def forward(self, inputs):
        n_input = np.prod(self.args.input_shape)
        x_ = inputs[:, :n_input].view(inputs.shape[0], *self.args.input_shape[1:])
        n = inputs[:, n_input:]
        z = self.encoder(x_)
        z1 = z + n
        x1 = self.decoder(z1)
        y = self.classifier(x1)
        return y

def get_clean_data(trainer, y, index=0):
    for i, (X, Y) in enumerate(trainer.test_loader):
        X = X.cpu()
        Y = Y.cpu()
        indices = np.where(Y == y)[0]
        if len(indices) > 0:
            j = indices[index % len(indices)]
            x = X[j][None]
            z = trainer.semcom.vae.encoder(x)
            x1 = trainer.semcom.vae.decoder(z)
            y_hat = trainer.semcom.pragmatic(x1)
            y_pred = y_hat.argmax(-1)
            if (y_pred == y).all():
                yield x, y_hat

def get_latent_feature_stats(trainer, args):
    Z = []
    for batch_id, (X, y) in enumerate(trainer.test_loader):
        X = X.to(args.device)
        y = y.to(args.device)
        z = trainer.semcom.vae.encoder(X)
        Z.append(z)
    Z = torch.cat(Z)
    power = (Z ** 2).sum(dim=1)
    z2_mean = power.mean().item()
    z2_ub = (power.mean() + 3 * power.std()).item()
    z_ub = (Z.mean(dim=0) + 3 * Z.std(dim=0)).detach().cpu().numpy()
    z_lb = (Z.mean(dim=0) - 3 * Z.std(dim=0)).detach().cpu().numpy()
    return z2_mean, z2_ub, z_ub, z_lb


class ZConstraint:

    def __init__(self, z2_ub, z_lb, z_ub, adv_snr_threshold):
        self.z2_ub = z2_ub
        self.z_lb = z_lb
        self.z_ub = z_ub
        self.adv_snr_threshold = adv_snr_threshold

    def constraint_fn(self, z):
        max_power = 10 ** (self.adv_snr_threshold / 10) * self.z2_ub
        if z ** 2 > max_power:
            return False
        if (z > self.z_ub).any():
            return False
        if (z < self.z_ub).any():
            return False
        return True

def prepare_split_benchmark(args):
    label = f'{args.experiment}_split_{args.adv_snr_threshold}_{args.n_latent}_{args.n_channel}_{args.r_lb}_{args.r_ub}'
    vnnlib_dir = os.path.join(args.vnnlib_dir, label)
    os.makedirs(vnnlib_dir, exist_ok=True)
    onnx_dir = os.path.join(args.onnx_dir, label)
    os.makedirs(onnx_dir, exist_ok=True)
    args.batch_size = 2
    args.device = 'cpu'
    net = SemCom(args)
    trainer = net.trainer
    z2_mean, z2_ub, z_ub, z_lb = get_latent_feature_stats(trainer, args)


    for x, _ in get_clean_data(trainer, 0):
        x = x.cpu()
        break

    z = net.encoder(x)
    rl = torch.ones_like(z.flatten(1)) * args.r_lb
    ru = torch.ones_like(z.flatten(1)) * args.r_ub

    path = os.path.join(onnx_dir, f'pgm.onnx')
    torch.onnx.export(
        trainer.generator,
        rl,
        path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )

    state_dict = {
        'z2_ub': z2_ub,
        'z_ub': z_ub,
        'z_lb': z_lb,
        'adv_snr_threshold': args.adv_snr_threshold,
    }
    path = os.path.join(onnx_dir, f'z_constraint.pt')
    torch.save(state_dict, path)

    solver, gurobi_vars, pre_relu_names, relu_names, lb, ub, nl, nu = build_milp_solver(
        net=net.generator.layers,
        input_lower=rl[0],
        input_upper=ru[0],
        refine=True,
        return_bounds=False,
        max_power= 10 ** (args.adv_snr_threshold / 10) * z2_ub,
        min_value=z_lb,
        max_value=z_ub,
    )

    path = os.path.join(vnnlib_dir, f'z_bound.pt')
    data = {
        'n_lb': lb,
        'n_ub': ub,
    }
    torch.save(data, path)

    print(f'{lb=}')
    print(f'{ub=}')
    print(f'{nl=}')
    print(f'{nu=}')


    for y in range(10):
        for j in range(args.n_sample_per_class):
            for X, y_pred in get_clean_data(trainer, y, index=j):
                break
            n_lb = nl.view(z.shape).to(z)
            n_ub = nu.view(z.shape).to(z)
            s_lb = torch.tensor(X.flatten(1) - args.linf_eps)
            s_ub = torch.tensor(X.flatten(1) + args.linf_eps)
            lb = torch.cat([s_lb, n_lb], dim=1)
            ub = torch.cat([s_ub, n_ub], dim=1)
            inputs = lb.clone()
            path = os.path.join(vnnlib_dir, f'{y}_{j}.vnnlib')
            simulator.util.spec.write_vnnlib(spec_path=path, data_lb=lb, data_ub=ub, prediction=y_pred)
    path = os.path.join(onnx_dir, f'net.onnx')
    torch.onnx.export(
        net,
        inputs,
        path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
