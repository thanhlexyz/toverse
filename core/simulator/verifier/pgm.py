from beartype import beartype
import torch
import time
import os

from simulator.verifier.util import check_solution
from simulator.model.gan import Generator

class Attacker:

    @beartype
    def __init__(self, net: torch.nn.Module, objectives, input_shape, args) -> None:
        self.G_a = Generator(args.n_latent, args.n_latent * 2).to(args.device)
        path     = os.path.join(args.checkpoint_dir, f'{args.experiment}_pgm',
                      f'generator_{args.adv_snr_threshold}_{args.n_latent}_{args.n_channel}.pth')
        assert os.path.exists(path)
        self.G_a.load_state_dict(torch.load(path))
        self.G_a.eval()
        self.net = net
        self.args = args
        self.objectives = objectives
        self.input_shape = input_shape
        
        path = os.path.join(args.onnx_dir, f'{args.experiment}_split_{args.adv_snr_threshold}_{args.n_latent}_{args.n_channel}_{args.r_lb}_{args.r_ub}/z_constraint.pt')
        assert os.path.exists(path)
        self.noise_constraint = torch.load(path, weights_only=False)


    def noise_constraint_fn(self, z: torch.Tensor) -> bool:
        max_power = 10 ** (self.noise_constraint['adv_snr_threshold'] / 10) * self.noise_constraint['z2_ub']
        assert z.shape[0] == 1
        if (z ** 2).sum().item() > max_power:
            return False
        if (z > torch.from_numpy(self.noise_constraint['z_ub']).to(z.device)).any():
            return False
        if (z < torch.from_numpy(self.noise_constraint['z_lb']).to(z.device)).any():
            return False
        return True
    
    @beartype
    def attack(self, timeout: float = 1.0) -> tuple[bool, torch.Tensor | None]:
        args = self.args
        input_lb = self.objectives.lower_bounds[0].view(self.input_shape).to(args.device)
        input_ub = self.objectives.upper_bounds[0].view(self.input_shape).to(args.device)
        
        r_lb = input_lb[..., -self.args.n_latent:]
        r_ub = input_ub[..., -self.args.n_latent:]
        
        x_lb = input_lb[..., :-self.args.n_latent]
        x_ub = input_ub[..., :-self.args.n_latent]
        
        start = time.time()
        while True:
            if time.time() - start > timeout:
                return False, None
            r = (r_ub - r_lb) * torch.rand_like(r_lb) + r_lb
            n = self.G_a(r)
            if not self.noise_constraint_fn(n):
                continue
            
            x = (x_ub - x_lb) * torch.rand_like(x_lb) + x_lb
            inputs = torch.cat([x, r], dim=1)
            is_attacked = check_solution(
                net=self.net,
                adv=inputs,
                cs=self.objectives.cs.to(args.device),
                rhs=self.objectives.rhs.to(args.device),
                data_min=input_lb,
                data_max=input_ub,
            )

            if is_attacked:
                return True, inputs

    def __repr__(self):
        return f'PGMAttacker(device={self.device})'
