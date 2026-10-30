from beartype import beartype
import random
import torch
import time
import os

from helper.network.onnx2pytorch import ConvertModel
from helper.spec.objective import DnfObjectives

from .general import attack
from helper.misc.check import check_solution
from helper.misc.logger import logger


class CPGDAttacker:

    @beartype
    def __init__(self: 'CPGDAttacker', net: ConvertModel | torch.nn.Module, objective: DnfObjectives, input_shape: tuple, device: str = 'cpu', noise_generator: ConvertModel | torch.nn.Module = None, noise_constraint: dict | None = None) -> None:
        self.net = net
        self.objective = objective
        self.input_shape = input_shape
        self.device = device
        self.seed = None
        self.noise_generator = noise_generator.to(device)
        self.noise_constraint = noise_constraint

    @beartype
    def manual_seed(self: 'CPGDAttacker', seed: int) -> None:
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    @beartype
    def noise_constraint_fn(self: 'CPGDAttacker', z: torch.Tensor) -> bool:
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
    def run(self: 'CPGDAttacker', iterations: int = 100000, restarts: int = 20, timeout: float = 2.0) -> tuple[bool, torch.Tensor | None]:
        
        assert self.noise_generator is not None
        assert self.noise_constraint is not None
        
        data_min = self.objective.lower_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        data_max = self.objective.upper_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        
        # assert torch.all(data_min <= data_max)
        # x = (data_min[:, 0] + data_max[:, 0]) / 2
        x = (data_max[:, 0] - data_min[:, 0]) * torch.rand(data_min[:, 0].shape, device=self.device) + data_min[:, 0]
        if os.environ.get('NEURALSAT_ASSERT'):
            assert torch.all(x <= data_max[:, 0])
            assert torch.all(x >= data_min[:, 0])
        
        cs = self.objective.cs.to(self.device)
        rhs = self.objective.rhs.to(self.device)
        
        print(f'Attacking CPGD F32 {iterations=} {restarts=} {timeout=} {cs.shape=} {rhs.shape=} {self.noise_generator=} {self.noise_constraint=}')
        self.noise_generator.num_z = len(self.noise_constraint['z_ub'])
        is_attacked, attack_images = attack(
            model=self.net,
            x=x.to(cs.dtype), 
            data_min=data_min,
            data_max=data_max,
            cs=cs,
            rhs=rhs,
            attack_iters=iterations, 
            num_restarts=restarts,
            timeout=timeout / 2.0,
            noise_generator=self.noise_generator,
            constraint_fn=self.noise_constraint_fn,
        )
    
        if is_attacked:
            with torch.no_grad():
                for i in range(attack_images.shape[1]): # restarts
                    for j in range(attack_images.shape[2]): # props
                        adv = attack_images[:, i, j]
                        if check_solution(self.net, adv, cs=cs[j], rhs=rhs[j], data_min=data_min[:, j], data_max=data_max[:, j]):
                            return True, adv
            logger.debug("[!] Invalid counter-example")
        
        self.net.to(cs.dtype)
        return False, None

    
    def __str__(self):
        return f'CPGDAttack(seed={self.seed}, device={self.device})'

