import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import lraspp_mobilenetv3_large


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def predict(image_path, result_dir):
    aux = False  # inference time not need aux_classifier
    classes = 1
    weights_path = "./save_weights/model_199.pth"
    img_path = image_path
    palette_path = "./palette.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = lraspp_mobilenetv3_large(num_classes=classes+1)

    # load weights
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    original_img = Image.open(img_path)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)
        # 从图像路径中提取文件名（不含扩展名）
        file_name = os.path.splitext(os.path.basename(image_path))[0]

        # 生成预测结果图像的保存路径
        output_path = os.path.join(result_dir, f"{file_name}_predicted.png")

        # 保存预测结果图像
        mask.save(output_path)


if __name__ == '__main__':
    # 定义图像目录路径
    image_dir = "test"
    result_dir = "test_result"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # 获取图像目录下所有jpg文件的路径
    jpg_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # 循环遍历每个jpg文件并调用predict函数
    for jpg_file in jpg_files:
        predict(jpg_file, result_dir)