"""
    官方demo
    添加详细注释，仅供学习
"""
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# 数据集预处理
"""
    标准化图像
    Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 50000张训练图片
# 第一次使用时要将download设置为True才会自动去下载数据集
train_set = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                         download=False, transform=transform)

# shuffle=True表示每次读取的图片是随机的，洗牌打乱、
# num_workers=？表示使用？线程读取，Windows下设置为0，否则报错
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                           shuffle=True, num_workers=0)

# 10000张验证图片
# 第一次使用时要将download设置为True才会自动去下载数据集
val_set = torchvision.datasets.CIFAR10(root='../../data', train=False,
                                       download=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=4,
                                         shuffle=False, num_workers=0)

val_data_iter = iter(val_loader)  # 转换为可迭代的迭代器
val_image, val_label = next(val_data_iter)  # 获取验证集的图像和对应的标签

# classes下标从0开始
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    # normalize：output = (input-0.5)/0.5
    # unnormalize：output = input * 0.5 + 0.5
    img = img * 0.5 + 0.5  # unnormalize 反标准化
    np_img = img.numpy()  # 转换为numpy格式
    # 转换为原始shape格式 [channel,height,width] -> [height,width,channel]
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


# 输出标签
# print(' '.join('%5s' % classes[val_label[i]] for i in range(4)))
# 显示图像
# imshow(torchvision.utils.make_grid(val_image[:4]))


net = LeNet()
'''
      Note that this case is equivalent to the combination of :class:`~torch.nn.LogSoftmax` and
      :class:`~torch.nn.NLLLoss`.
      已经包含了softmax和交叉熵损失函数
      所以在定义网络的时候，最后没有加softmax
'''
loss_function = nn.CrossEntropyLoss()
# 设置优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)
# 开始训练，一共5轮
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0  # 累加训练集的损失
    for step, data in enumerate(train_loader, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        # 如果不清空历史梯度，会对计算的历史梯度进行累加（通过这个特性可以能够变相实现一个很大的batch数值的训练）
        optimizer.zero_grad()
        # 训练过程
        outputs = net(inputs)  # 正向传播
        loss = loss_function(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:    # print every 500 mini-batches
            # with 是一个上下文管理器，可以将一些操作在一个代码块中执行，
            # 在接下来的计算中，不要计算每个节点的误差损失梯度
            # 如果删掉这句话，在验证的过程中，也会计算误差损失梯度，会导致内存爆炸
            # 而这种资源的浪费是没有必要的
            with torch.no_grad():
                outputs = net(val_image)  # [batch, 10]
                # 第0个纬度是batch，第1个维度是10个节点
                # outputs指网络的输出，在第一个维度中寻找10个节点的最大值，[1]指得是索引
                predict_y = torch.max(outputs, dim=1)[1]
                # eq(predict_y, val_label)相等返回1，否则返回0，通过sum()求和，通过item()转换为python的float类型
                # 除以val_label.size(0)是为了计算准确率，因为val_label.size(0)是验证集的样本
                accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')

save_path = './Lenet.pth'
# 保存模型，pth
torch.save(net.state_dict(), save_path)
