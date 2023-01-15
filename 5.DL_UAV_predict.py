# Deep learning-based UAV-RGB coastal wetland vegetation classification
# 1)cfg, Set the parameters and paths;
# 2)train, Build the dataset and train the DL model;
# 3)evalution, Evaluation of the model;
# 4)predict, Read model weights and make predictions (vegetation classification).
#
# Time: 2022-10-20
# 
# E-mail:19210700109@fudan.edu.cn
# --------------------------------------------------


import cv2
import time
import numpy as np
import RSjunyi.rs as grs
import rasterio
import torch.nn as nn
import torch
import torch.nn.functional
import torch.utils.data as D
import torchvision
from Dataprocessing import *
from cfg import *
from model import *

from torchvision import transforms as T
import segmentation_models_pytorch as smp
from tqdm import tqdm

as_tensor = T.Compose([
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@torch.no_grad()
def predict(model, img_path, predict_path, name, stride=128, img_h=256, img_w=256):
    start = time.time()

    model = model.eval()
    print('predicting....................................')

    src = rs.open(img_path)

    img = grs.img_read(img_path, info=False)
    img = img[:, :, 0:3]
    # print(img.shape)
    h, w, band = img.shape
    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    # padding 3-d
    padding_img = np.zeros((padding_h, padding_w, band), dtype=np.uint8)
    padding_img[0:h, 0:w, :] = img[:, :, :]
    # mask of predict 2-d
    mask = np.zeros((padding_h, padding_w))

    # predict for each image
    for i in range(padding_h // stride):
        for j in range(padding_w // stride):
            print('step raw:{}, colimumn:{}'.format(i, j))
            crop = padding_img[i * stride:i * stride + img_h,
                   j * stride: j * stride + img_w, :]
            if crop.shape != (img_h, img_w, band):
                print('invalid size, crop.shape:{}, img.shape:{}'.format(crop.shape, (img_h, img_w, band)))
                continue
            crop = as_tensor(crop)
            crop = crop.unsqueeze(0)

            model.to(DEVICE)
            crop = crop.to(DEVICE)

            model.eval()

            output = model(crop).cpu()

            predict = output[0].argmax(0)
            mask[i * stride:i * stride + img_h, j * stride: j * stride + img_w] = predict[:, :]

    predict_img = mask[0:h, 0:w]
    print(predict_img.shape)
    path = predict_path.format(name, time.strftime("%m%d-%H%M", time.localtime()))
    rs_img = rasterio.open(img_path)

    with rasterio.Env():

        profile = rs_img.profile
        dtype = profile['dtype']
        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(
            count=1, compress='lzw') 

        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(predict_img.astype(dtype), 1)

    end = time.time()
    print('Finished, time spent:{:.2f}s'.format(end - start))
    return predict_img


if __name__ == '__main__':

    DeepLabV3p = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet',
                              encoder_output_stride=16, decoder_channels=256, decoder_atrous_rates=(12, 24, 36),
                              in_channels=3, classes=5, activation=None, upsampling=4, aux_params=None)


    DeepLabV3p.load_state_dict(torch.load('/media/gentry/数据分区/DL核心数据2022/model_pth/PSPNet20220908_label2_0909.pth'
, map_location='cpu'))

    predict(DeepLabV3p, pt_2, predict_path, 'DeepLabV3p')
    predict(DeepLabV3p, pt_7, predict_path, 'DeepLabV3p')

    print('ok')
