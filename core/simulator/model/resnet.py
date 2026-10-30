import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn

try:
    from timm.models.registry import register_model
except ModuleNotFoundError:
    def register_model(func):
        return func


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class PaddingLayer(nn.Module):

    def __init__(self, padding):
        super(PaddingLayer, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.padding, self.padding), "constant", 0)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, option='A', expansion=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.expansion = expansion

        self.shortcut = nn.Identity()
        if (stride != 1 or in_planes != planes) and option == 'A':
            self.shortcut = PaddingLayer(planes // 4)
        elif option == 'B':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlockBN(nn.Module):

    def __init__(self, in_planes, planes, stride=1, option='A', expansion=1):
        super(BasicBlockBN, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(planes)
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(planes)
        )
        self.expansion = expansion

        self.shortcut = nn.Identity()
        if (stride != 1 or in_planes != planes) and option == 'A':
            self.shortcut = PaddingLayer(planes // 4)
        elif option == 'B':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.seq1(x))
        out = self.seq2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreLayer(nn.Module):

    def __init__(self, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, planes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        return self.conv1(x)


class PostLayer(nn.Module):

    def __init__(self, dim, num_classes=10):
        super().__init__()
        self.linear2 = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = x.mean(dim=[2, 3])
        x = self.linear2(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, option='B'):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.planes = 32
        pre_layer = PreLayer(self.in_planes)
        assert len(num_blocks)
        layers = self._make_layer(block, self.planes, num_blocks[0], stride=1, expansion=1, option=option)
        layers[0] = nn.Sequential(pre_layer, layers[0])
        if len(num_blocks) > 1:
            layers += self._make_layer(block, self.planes, num_blocks[1], stride=2, expansion=1, option=option)
        if len(num_blocks) > 2:
            layers += self._make_layer(block, self.planes, num_blocks[2], stride=2, expansion=1, option=option)
        layers[-1] = nn.Sequential(layers[-1], PostLayer(dim=self.planes, num_classes=num_classes))

        self.layers = nn.ModuleList(layers)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option, expansion):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=stride,
                    option=option,
                    expansion=expansion,
                )
            )
            self.in_planes = planes * expansion
        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
