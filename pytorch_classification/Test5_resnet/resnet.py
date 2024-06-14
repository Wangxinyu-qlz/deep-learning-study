"""
    ResNet 18 34 50 101 152模型搭建
"""
import os
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # output = (input - kernel + 2 * padding)/stride + 1
    # Stride=1时，output = (input - 3 + 2 * 1)/1 + 1 =input
    # Stride=1时，output = input/2+0.5 = input/2(向下取整)
    # bias = False 因为BN层之前使用bias没有用，不需要
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# Batch Normalization
# BN层的目的是使一批（batch）的feature map满足均值为0，方差为1的分布
# 18 34层对应的残差结构
class BasicBlock(nn.Module):
    expansion = 1  # 残差结构中的 主分支中 卷积核个数 有没有变化

    # downsample:对应虚线残差结构
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        #  BN层放在卷积层和激活函数之间
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # 残差结构中的第二个卷积层之后没有ReLu
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 捷径分支上的输出
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果没有下采样：捷径分支的输出就是x，如果有下采样：捷径分支的输出就是x下采样之后的值
        if self.downsample is not None:
            identity = self.downsample(x)

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
    # 第3层的卷积核的个数是第1、2层的4倍
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # ResNeXt的分组卷积
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # 以下结构中，ResNet:out_channels=out_channel    ResNeXt:out_channels=width
        # conv3中的out_channels不变 始终为out_channel*self.expansion
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        # -----------------------------------------
        # 这里的stride为2
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        # -----------------------------------------
        # 卷积核的个数是第1、2层的4倍
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 如果没有下采样：捷径分支的输出就是x，如果有下采样：捷径分支的输出就是x下采样之后的值
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block,  # 选择不同的残差结构，18、34：BasicBlock  50、101、152：Bottleneck
                 layers,  # 残差结构的数量，list变量  ResNet50:[3,4,6,3]
                 num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # 输入特征矩阵的深度 所有网络(34/50/101/152)通用的
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.block = block
        self.groups = groups
        self.base_width = width_per_group

        # output = (input - kernel_size + padding * 2) / stride + 1
        # 经过conv1之后，尺寸缩减为原来的一半
        # 224,224,3 -> 112,112,64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 输出特征矩阵的height和width减少到输入特征矩阵的一半->stride=2，
        # 和输入特征矩阵的size相同->padding=1
        # 112,112,64 -> 56,56,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 56,56,64 -> 56,56,256
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 56,56,256 -> 28,28,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        # 28,28,512 -> 14,14,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # 14,14,1024 -> 7,7,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # 自适应 平均池化下采样
        # 对于输入的任意尺寸的特征矩阵，输出的特征矩阵的宽和高都是1
        # 7,7,2048 -> 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)

        # 2048 -> num_classes
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 卷积层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    # channel为残差结构中第一层的卷积核的个数
    # block_num为该层中的残差结构的个数
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 第1层的stride=1
        #   对于18 34层网络，expansion=1，第二个条件不成立，跳过downsample
        #   对于50 101 152层的网络，expansion=4，第二个条件成立，执行downsample
        # 第2、3、4层的stride=2，18/34/50/101/152层的网络都会执行downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # Conv_block
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        #  对于18/34层网络，输出channel==输入channel
        #  对于50/101/152层的网络，输出channel=输入channel * 4
        self.inplanes = planes * block.expansion
        # 通过循环将不需要下采样的（实线）残差结构添加到layers中
        # 从1开始，即第二层
        for _ in range(1, blocks):
            # identity_block
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        # 通过非关键字参数的形式生成网络结构
        # *layers中的*表示非关键字
        # 函数被调用的时候，使用星号 * 解包一个可迭代对象(元组)作为函数的参数。
        # 字典对象，可以使用两个星号 ** ，解包之后将作为关键字参数传递给函数
        # 函数传递参数的方式位置参数*args（positional argument） 关键词参数**kwargs（keyword argument）
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # Conv_2
        x = self.layer2(x)  # Conv_3
        x = self.layer3(x)  # Conv_4
        x = self.layer4(x)  # Conv_5

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def freeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = False

    def Unfreeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = True


def get_resnet(model_name, pretrained=False, progress=True, num_classes=1000):
    """
    根据给定的模型名称返回一个预训练的ResNet模型。

    参数:
    model_name (str): 模型名称，可以是'resnet18'、'resnet34'、'resnet50'、'resnet101'或'resnet152'。
    pretrained (bool): 如果为True，则返回预训练的模型；否则返回一个随机初始化的模型。
    progress (bool): 如果为True，则显示下载进度条；否则不显示。
    num_classes (int): 模型输出类别数，默认为1000，可以根据实际需求调整。

    返回:
    torchvision.models.ResNet: 创建的ResNet模型。
    """
    # 确保模型名称在可用模型列表中
    assert model_name in model_urls, f"Warning: model {model_name} not in model_urls dict!"

    # 模型配置字典，映射模型名称到其对应的BasicBlock或Bottleneck和层配置
    model_config = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3])
    }

    # 根据模型名称获取对应的Block类型和层配置
    block, layers = model_config[model_name]
    # 创建ResNet模型实例
    model = ResNet(block, layers, num_classes=num_classes)

    # 如果需要预训练模型
    if pretrained:
        # 定义模型存储目录
        model_dir = "../model_data"
        # 如果目录不存在，则创建目录
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # 从URL加载预训练模型的状态字典
        state_dict = load_state_dict_from_url(model_urls[model_name], model_dir=model_dir, progress=progress)

        # 如果指定的类别数与预训练模型的不同，则调整模型的最后全连接层
        if num_classes != 1000:
            # 删除预训练模型中与新类别数不匹配的fc层参数
            for k in list(state_dict.keys()):
                if k.startswith('fc'):
                    del state_dict[k]
            # 创建新的fc层，适应新的类别数
            model.fc = nn.Linear(512 * block.expansion, num_classes)

        # 加载调整后的预训练模型参数
        model.load_state_dict(state_dict, strict=False)

    # 如果类别数与预训练模型的不同，再次调整模型的最后全连接层（确保即使pretrained为False时也能正确处理）
    if num_classes != 1000:
        model.fc = nn.Linear(512 * block.expansion, num_classes)

    # 返回构建或调整好的模型
    return model


if __name__ == '__main__':
    input1 = torch.rand([32, 3, 224, 224])
    model = get_resnet(model_name='resnet34', pretrained=True, num_classes=1000)
    # print(model)
    output = model(input1)
    # print(output.shape)

    # 可视化网络结构
    from torchview import draw_graph

    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    model_graph = draw_graph(model, input_size=input1.shape, depth=2, graph_dir='TB', expand_nested=True,
                             save_graph=True, filename="resnet18", directory=".")
    model_graph.visual_graph
