import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

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
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
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
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def get_vgg(model_name="vgg16", pretrained=False, progress=True, num_classes=1000, init_weights=True):
    """
    根据给定的模型名称初始化VGG模型。

    参数:
    model_name (str): 模型名称，默认为"vgg16"。应与配置文件中的模型名称匹配。
    pretrained (bool): 是否使用预训练的权重初始化模型。默认为False。
    progress (bool): 是否显示下载模型权重的进度条。默认为True。
    num_classes (int): 模型输出类别数。默认为1000，适用于ImageNet分类任务。
    init_weights (bool): 是否初始化模型的权重。默认为True。

    返回:
    VGG模型实例。
    """
    assert model_name in cfgs, f"Warning: model number {model_name} not in cfgs dict!"
    cfg = cfgs[model_name]
    # 初始化VGG模型，包括特征提取部分和分类器
    model = VGG(make_features(cfg), num_classes=num_classes, init_weights=init_weights)

    if pretrained:
        # 定义模型权重存储的目录
        model_dir = "../model_data"
        # 如果目录不存在，则创建
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # 从URL加载预训练的模型权重
        state_dict = load_state_dict_from_url(model_urls[model_name], model_dir=model_dir, progress=progress)

        # 定义需要从预训练权重中删除的键，因为这些部分需要根据新任务进行重新训练
        classifier_keys = ['classifier.0.weight', 'classifier.0.bias',
                           'classifier.3.weight', 'classifier.3.bias',
                           'classifier.6.weight', 'classifier.6.bias']

        # 删除预训练权重中与当前任务不匹配的部分
        for key in classifier_keys:
            if key in state_dict:
                del state_dict[key]

        # 加载预训练权重到模型，不严格要求所有权重匹配
        model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == '__main__':
    # TODO 网络结构可视化
    from torchview import draw_graph
    import os
    import torch

    x = torch.randn(32, 3, 224, 224)
    model = get_vgg(model_name="vgg13", pretrained=True, num_classes=1000, init_weights=True)
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    model_graph = draw_graph(model, input_size=x.shape, depth=3, graph_dir='TB', expand_nested=True,
                             save_graph=True, filename="vgg16", directory=".")
    model_graph.visual_graph
    print("网络结构已保存")
