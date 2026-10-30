from torch import nn
import torch

from ..utils import PRINT_DEBUG

class Gather(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        self.selection = [slice(None) for _ in range(dim)]
        super().__init__()

    def forward(self, data: torch.Tensor, indices: torch.Tensor):
        if PRINT_DEBUG:
            print('GATHER:', data.shape, data.is_floating_point())
                
            print('\t- self.dim:', self.dim)
            print('\t- indices:', indices)
            
        if data.is_floating_point():
            dim = max(self.dim, 1)
        else:
            dim = self.dim
            
        
        if indices.numel() == 1 and indices == -1:
            indices = torch.tensor(data.shape[dim] - 1, device=data.device)

        
        final = torch.index_select(data, dim=dim, index=indices).squeeze(dim)
            
        if PRINT_DEBUG:
            print('\t- dim:', dim)
            print('\t- final:', final)
            print('\t- final:', final.shape)
            assert final.numel() > 0
        return final
