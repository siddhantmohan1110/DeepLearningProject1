'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import params

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = dropout

        if self.dropout:
            self.dropoutlayer = nn.Dropout(p=0.3) #added dropout in each residual block
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            out = self.dropoutlayer(out) #added dropout after computation of ReLU
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ModResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, type=1, dropout=False):
        super(ModResNet, self).__init__()
        self.in_planes = 64
        if type==1:
            channels = [64, 128, 192, 256] #type-1 ModResNet (4.1M params)
        elif type==2:
            channels = [64, 128, 208, 288] #type-2 ModResNet (4.9M params)
        else:
            raise ValueError('Only 1 or 2 are valid types of ModResNet')
        
        self.dropout = dropout

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], stride=1, dropout=self.dropout)
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], stride=2, dropout=self.dropout)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], stride=2, dropout=self.dropout)
        self.layer4 = self._make_layer(block, channels[3], num_blocks[3], stride=2, dropout=self.dropout)
        if dropout:
            self.dropoutlayer = nn.Dropout(p=0.3) #added dropout just before linear layer
        self.linear = nn.Linear(channels[3]*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if self.dropout:
            out = self.dropoutlayer(out)
        out = self.linear(out)
        return out


def ModResNet18(type, dropout):
    return ModResNet(BasicBlock, [2, 2, 2, 2], type=type, dropout=dropout)


def test():
    net = ModResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
