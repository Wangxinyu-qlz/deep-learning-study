import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import torchvision.models.resnet
from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     # TODO 官方标准化参数
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),  # 长宽比固定，最小边缩放到256
                                   transforms.CenterCrop(224),  # 中心裁剪
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # # TODO 迁移学习 1.官方提供的载入预训练模型的方法
    # net = resnet34()  # 没有传入num_classes参数，实例化之后，全连接层有1000个节点（1000分类）
    # model_weight_path = "./resnet34-pre.pth"
    # missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
    # in_channel = net.fc.in_features  # 输入特征矩阵的深度
    # net.fc = nn.Linear(in_channel, 5)  # 全连接层，重新复制全连接层
    # net.to(device)

    # # TODO 迁移学习 2.第二种方法
    # net = resnet34(num_classes=5)  # 传入num_classes参数，实例化之后，全连接层有5个节点（5分类）
    # model_weight_path = "./resnet34-pre.pth"
    # # (1)通过torch.load()将预训练参数载入内存，并未载入模型中，得到的是一个字典
    # pth_dict = torch.load(model_weight_path)
    # # (2)将全连接层的参数删除
    # new_state_dict = {}
    # for key, value in pth_dict.items():
    #     if not key.startswith('fc.'):  # 假设全连接层的命名是'fc'
    #         new_state_dict[key] = value
    # # (3)将预训练参数载入模型
    # net.load_state_dict(new_state_dict)

    net = resnet34()  # 没有传入num_classes参数，实例化之后，全连接层有1000个节点（1000分类）
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # TODO 改一下map_location???
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    # TODO 冻结所有网络的权重，目的是只训练最后一层
    # for param in net.parameters():
    #     param.requires_grad = False
    in_channel = net.fc.in_features  # 输入特征矩阵的深度
    net.fc = nn.Linear(in_channel, 5)  # 全连接层，重新复制全连接层
    net.to(device)

    # 如果不是用迁移学习，使用以下代码
    # net = resnet34(num_classes=num_classes)
    # net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 3
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # TODO train 控制NB层的状态
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # TODO validate 控制NB层的状态
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
