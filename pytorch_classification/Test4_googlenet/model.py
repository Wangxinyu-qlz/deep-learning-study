import torch.nn as nn
import torch
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        # 是否使用辅助分类器
        self.aux_logits = aux_logits

        # 64？：看论文中的参数表格
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # ceil_mode=True小数向上取整  ceil_mode=False小数向下取整
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 这里因为LocalRespNorm层没有什么效果，所以丢弃了
        # nn.LocalResponseNorm()

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # TODO why256?
        #  256 = 64+128+32+32(在inception模块)
        #         outputs = [branch1, branch2, branch3, branch4]
        #         # 对四个分支的结果进行合并，需要合并的维度为channels，在深度上进行拼接，所以dim=1
        #         # [batch, channels, height, width]
        #         return torch.cat(outputs, 1)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        # 平均池化下采样
        # 自适应池化下采样，对于输入的任意尺寸(h,w)的特征矩阵，输出的特征矩阵总是(1,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        # 展平后的结点个数是1024，输出的个数是num_classes
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        # 训练模型使用辅助分类器，验证模式不使用
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        # 全连接
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# inception模块
class Inception(nn.Module):
    """
        in_channels: 输入通道数
        1.  ch1x1: 1x1卷积核的个数
        2.  ch3x3red: 1x1卷积核的个数 降维
            ch3x3: 3x3卷积核的个数
        3.  ch5x5red: 1x1卷积核的个数 降维
            ch5x5: 5x5卷积核的个数
        4.  (max_pooling)
            pool_proj: 1x1卷积核的个数
        1、2、3、4为并行结构
    """
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        # 每个分支所得的特征矩阵的高和宽应当相等
        # 第一个分支
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)  # stride=1(default)下同

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            # TODO padding=1以保证输出大小等于输入大小 output = (intput - kernel_size + 2×padding) / stride + 1
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            # 在官方的实现中，其实是3x3的kernel并不是5x5，这里我也懒得改了，具体可以参考下面的issue
            # 官方实现中加入了BN层，并使用两个3×3 kernel代替一个5×5 kernel
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        # 对四个分支的结果进行合并，需要合并的维度为channels，在深度上进行拼接，所以dim=1
        # [batch, channels, height, width]
        return torch.cat(outputs, 1)


# 辅助分类器模板 auxiliary classifier template
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        # 全连接层
        # 128 * 4 * 4 = 2048
        # 这里的ReLU层在forward函数中
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # N: batch_size
        # aux1输入的特征维度: N x 512 x 14 x 14, aux2输入的特征维度: N x 528 x 14 x 14
        x = self.averagePool(x)

        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)

        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)

        # TODO 原文中使用的0.7，但是工程实践中0.5最好，具体根据自己的任务进行调整
        x = F.dropout(x, 0.5, training=self.training)

        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # 实例化一个模型后，通过model.train()和model.eval()控制模型的状态
        # model.train()->self.training=True
        # model.train()->self.training=False
        x = F.dropout(x, 0.5, training=self.training)

        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


# 卷积模板（Conv2d + ReLU），精简代码
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
