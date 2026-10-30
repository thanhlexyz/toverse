from torch import nn
import warnings
import torch

from ..utils import is_constant, get_selection, PRINT_DEBUG
from .base import Operator


class Add(Operator):
    
    def __init__(self, input_shape=None, input_indices=None, feature_dim=1):
        self.input_shape = input_shape
        self.input_indices = input_indices
        self.feature_dim = feature_dim
        self.out = None

        if input_shape and input_indices:
            self.out = torch.zeros(input_shape)

        super().__init__()

    def forward(self, *input):
        if PRINT_DEBUG:
            print('ADD:', [_.shape for _ in input])
        
        if self.input_indices:
            out = self.out * 0
            for inp, idx in zip(input, self.input_indices):
                selection = get_selection(idx, self.feature_dim)
                out[selection] += inp
            return out

        if is_constant(input[0]):
            input = sorted(input, key=lambda x: -len(x.shape))
        elif all(x == 1 for x in input[0].shape):
            input = sorted(input, key=lambda x: -sum(x.shape))

        out = input[0].clone()
        if PRINT_DEBUG:
            print('\t- out:', out.shape)
            print('\t- inputs:', [_.shape for _ in input])
            print('\t- inputs:', input[-1])
        
            
        for inp in input[1:]:
            out += inp
        return out

    def set_input_indices(self, input):
        assert isinstance(input, (list, tuple))

        if sum(is_constant(inp) for inp in input) >= len(input) - 1:
            return

        input_shape = input[0].shape
        if not all(input_shape == inp.shape for inp in input[1:]):
            warnings.warn("Addition might be corrupted.", RuntimeWarning)
        assert all(
            is_constant(inp) or input_shape[-1] == inp.shape[-1] for inp in input
        )

        while self.feature_dim >= len(input_shape):
            self.feature_dim -= 1
        axis = self.get_axis(input_shape, self.feature_dim)

        input_indices = []
        for inp in input:
            mask = inp != 0
            if len(inp.shape) > 1:
                s = mask.sum(axis=tuple(axis))
                seq = torch.arange(len(s))
                if torch.equal(s, seq) or torch.equal(s.flip(0), seq):
                    return
                mask = s != 0
            (non_zeros,) = torch.where(mask)
            input_indices.append(non_zeros)

        if all(len(i) == len(mask) for i in input_indices):
            return

        unique_indices = torch.cat(input_indices).unique()
        input_shape = list(input[0].shape)
        input_shape[self.feature_dim] = len(unique_indices)

        _, input_indices[0] = torch.where(
            input_indices[0][:, None] == unique_indices[None]
        )
        _, input_indices[1] = torch.where(
            input_indices[1][:, None] == unique_indices[None]
        )

        self.input_indices = input_indices
        self.input_shape = tuple(input_shape)
        self.out = nn.Parameter(
            torch.zeros(self.input_shape, device=input[0].device, dtype=input[0].dtype),
            requires_grad=False,
        )

    def __str__(self):
        if self.input_indices:
            return "Add({}, {}, {})".format(
                tuple(self.input_shape),
                len(self.input_indices[0]),
                len(self.input_indices[1]),
            )
        else:
            return "Add(None, None)"
