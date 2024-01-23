"""
    ResNet 34 50 101模型搭建
"""
import torch.nn as nn
import torch


# TODO Batch Normalization
#  BN层的目的是使一批（batch）的feature map满足均值为0，方差为1的分布
# 18 34层对应的残差结构
class BasicBlock(nn.Module):
    expansion = 1  # 残差结构中的 主分支中 卷积核个数 有没有变化

    # downsample:对应虚线残差结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # output = (input - kernel + 2 * padding)/stride + 1
        # Stride=1时，output = (input - 3 + 2 * 1)/1 + 1 =input
        # Stride=1时，output = input/2+0.5 = input/2(向下取整)
        # TODO bias = False 因为BN层之前使用bias没有用，不需要
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # TODO training=self.training?
        #  BN层放在卷积层和激活函数之间
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        # 残差结构中的第二个卷积层之后没有ReLu
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample

    def forward(self, x):
        # 捷径分支上的输出
        identity = x
        # 如果没有下采样：捷径分支的输出就是x，如果有下采样：捷径分支的输出就是x下采样之后的值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# 50 101 152层对应的残差结构
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4  # 第3层的卷积核的个数是第1、2层的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        # TODO ResNeXt的分组卷积
        width = int(out_channel * (width_per_group / 64.)) * groups

        # TODO 以下结构中，ResNet:out_channels=out_channel    ResNeXt:out_channels=width
        #  conv3中的out_channels不变 始终为out_channel*self.expansion
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,  # 选择不同的残差结构，18、34：BasicBlock  50、101、152：Bottleneck
                 blocks_num,  # 残差结构的数量，list变量  ResNet50:[3,4,6,3]
                 num_classes=1000,
                 include_top=True,  # 为了能够在ResNet基础上搭建更复杂的网络(作为主干网络，舍弃全连接层)
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 输入特征矩阵的深度 所有网络(34/50/101/152)通用的

        self.groups = groups
        self.width_per_group = width_per_group
        # output = (input - kernel_size + padding * 2) / stride + 1
        # 经过conv1之后，尺寸缩减为原来的一半
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        # 输出特征矩阵的height和width  减少到输入特征矩阵的一半->stride=2，和输入特征矩阵的size相同->padding=1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            # 自适应 平均池化下采样
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 卷积层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # channel为残差结构中第一层的卷积核的个数
    # block_num为该层中的残差结构的个数
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 第1层的stride=1   对于18 34层网络，expansion=4，跳过downsample | 对于50 101 152层的网络，expansion=4，执行downsample
        # 第2、3、4层的stride=2，对于18/34/50...网络都会执行downsample
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = [block(self.in_channel,
                        channel,
                        downsample=downsample,
                        stride=stride,
                        groups=self.groups,
                        width_per_group=self.width_per_group)]
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
