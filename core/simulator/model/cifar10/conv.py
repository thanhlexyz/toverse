import torch.nn as nn

class Conv(nn.Conv2d):
    def __init__(self, ci, co, **kw):
        super().__init__(ci, co, **{'kernel_size': 3, 'padding': 1, 'bias': False, **kw})
