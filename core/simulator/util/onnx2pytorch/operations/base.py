from torch import nn
from abc import ABC


class Operator(nn.Module, ABC):
    
    @staticmethod
    def get_axis(input_shape, input_feature_axis):
        if input_feature_axis < 0:
            input_feature_axis += len(input_shape)
        axis = set(range(len(input_shape))) - {input_feature_axis}
        return tuple(axis)


class OperatorWrapper(Operator, ABC):
    
    def __init__(self, op):
        self.forward = op
        self.__class__.__name__ = op.__name__
        super().__init__()
