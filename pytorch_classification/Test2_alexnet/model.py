import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 通过nn.Sequential()精简代码
        self.features = nn.Sequential(
            # TODO channels的设置
            #     多通道卷积过程，是输入一张三通道的图片，这时有多个卷积核进行卷积，并且每个卷积核都有三通道，分别对这张输入图片的三通道进行卷积操作。
            #     每个卷积核，分别输出三个通道，这三个通道进行求和，得到一个feature_map，有多少个卷积核，就有多少个feature_map
            # output = (intput - kernel_size + padding * 2) / stride + 1
            #        = (224 - 11 + 2 * 2) / 4 + 1 = 54.25 + 1 = 55
            nn.Conv2d(3, 128, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            # inplace=True 较少内存占用
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]

            nn.Conv2d(128, 256, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]

            nn.Conv2d(256, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            # 将全连接层的节点随机失活，达到防止过拟合的作用，某个节点随机失活的概率为0.5
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        # 初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # 展平，从维度1开始，即不动batch，只进行[channel, height, width]的展平
        # 原始维度：[2, 2, 2] 第0维：最外层中括号中有几个元素：[[1, 2],[3, 4]]和[[5, 6], [7, 8]] 第一维：[1, 2]和[3, 4] 第二维：1和2
        # >> > t = torch.tensor([[[1, 2],
        #                         [3, 4]],
        #                        [[5, 6],
        #                         [7, 8]]])
        # 将所有维度展平 0~2维度 --> [8]
        # >> > torch.flatten(t)
        # tensor([1, 2, 3, 4, 5, 6, 7, 8])
        # 将 1~2 维展平 --> [2, 4]
        # >> > torch.flatten(t, start_dim=1)
        # tensor([[1, 2, 3, 4],
        #         [5, 6, 7, 8]])
        # 也可以使用view()函数进行展平
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 遍历网络中的所有模块
        for m in self.modules():
            # 判断当前层结构的类别
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # 对权重进行初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
