import torch.nn as nn
import numpy as np
import simulator
import torch
import os

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

    def forward_clean(self, inputs):
        n_input = np.prod(self.args.input_shape)
        x_ = inputs[:, :n_input].view(inputs.shape[0], *self.args.input_shape[1:])
        z = self.encoder(x_)
        z1 = z
        x1 = self.decoder(z1)
        y = self.classifier(x1)
        return y

    def forward(self, inputs):
        n_input = np.prod(self.args.input_shape)
        x_ = inputs[:, :n_input].view(inputs.shape[0], *self.args.input_shape[1:])
        r = inputs[:, n_input:]
        n = self.generator(r)
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

def prepare_e2e_benchmark(args):
    label = f'{args.experiment}_e2e_{args.adv_snr_threshold}_{args.n_latent}_{args.n_channel}_{args.r_lb}_{args.r_ub}'
    vnnlib_dir = os.path.join(args.vnnlib_dir, label)
    os.makedirs(vnnlib_dir, exist_ok=True)
    onnx_dir = os.path.join(args.onnx_dir, label)
    os.makedirs(onnx_dir, exist_ok=True)
    args.device = 'cpu'
    net = SemCom(args)
    trainer = net.trainer
    for x, _ in get_clean_data(trainer, 0):
        x = x.cpu()
        break
    for x, _ in get_clean_data(trainer, 0):
        x = x.cpu()
        break
    z = net.encoder(x)
    r_lb = torch.full_like(z, args.r_lb)
    r_ub = torch.full_like(z, args.r_ub)
    for y in range(10):
        for j in range(args.n_sample_per_class):
            for X, y_pred in get_clean_data(trainer, y, index=j):
                break
            s_lb = torch.tensor(X.flatten(1) - args.linf_eps)
            s_ub = torch.tensor(X.flatten(1) + args.linf_eps)
            lb = torch.cat([s_lb, r_lb], dim=1)
            ub = torch.cat([s_ub, r_ub], dim=1)
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
