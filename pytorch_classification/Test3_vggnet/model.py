import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Linear(512*7*7, 4096),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # nn.Linear(4096, 4096),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # nn.Linear(4096, num_classes)
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    # 通过非关键字参数的形式生成网络结构
    # *layers中的*表示非关键字
    # 函数被调用的时候，使用星号 * 解包一个可迭代对象(元组)作为函数的参数。
    # 字典对象，可以使用两个星号 ** ，解包之后将作为关键字参数传递给函数
    # 函数传递参数的方式位置参数*args（positional argument） 关键词参数**kwargs（keyword argument）
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 实例化配置模型
def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    # 通过给定的key得到配置列表（网络的结构参数列表）
    cfg = cfgs[model_name]
    # 实例化VGG网络，
    model = VGG(make_features(cfg), **kwargs)
    return model


config = {"num_classes": 1000, "init_weights": True}
vgg_model = vgg(model_name="vgg13", **config)


if __name__ == '__main__':
    # TODO 网络结构可视化
    from torchview import draw_graph
    import os
    import torch
    x = torch.randn(32, 3, 224, 224)
    model = vgg()
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    model_graph = draw_graph(model, input_size=x.shape, depth=3, graph_dir='TB', expand_nested=True,
                             save_graph=True, filename="vgg16", directory=".")
    model_graph.visual_graph
    print("网络结构已保存")
