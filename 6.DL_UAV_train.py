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


from comet_ml import Experiment
from evalution import *
from cfg import *
from Dataprocessing import make_grid
import albumentations as A
import rasterio as rs
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import random
from rasterio.windows import Window
import pathlib
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision
from torchvision import transforms as T
import segmentation_models_pytorch as smp
from model import*


trfm = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        A.ColorJitter(brightness=0.07, contrast=0.07,
                      saturation=0.1, hue=0.1, always_apply=False, p=0.3),
    ], p=0.3),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
    ], p=0.0),
    A.ShiftScaleRotate(),
])

def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seeds()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class VegDataset(D.Dataset):
    def __init__(self, window=256, overlap=0, transform=trfm):
        # self.path = pathlib.Path(root_dir)
        self.transform = transform
        self.window = window
        self.overlap = overlap
        self.images, self.labels = [], []
        self.build_slices()
        self.len = len(self.images)
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    def build_slices(self):
        list_img = os.listdir(imgs_path)
        list_label = os.listdir(labels_path)
        list_img.sort()
        list_label.sort()
        for i, j in zip(list_img, list_label):
            img_path = '{}'.format(os.path.join(imgs_path, i))
            label_path = '{}'.format(os.path.join(labels_path, j))
            img_data = rs.open(img_path)
            label_data = rs.open(label_path)
            clip_index = make_grid(img_data.shape, window=self.window, min_overlap=self.overlap)
            for slc in tqdm(clip_index):
                x1, x2, y1, y2 = slc

                img = img_data.read([1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2)))
                img = np.transpose(img, [1, 2, 0])
                img = img.astype(np.uint8)

                label = label_data.read(window=Window.from_slices((x1, x2), (y1, y2)))
                label = np.transpose(label, [1, 2, 0])
                # label.shape: (h, w, 1), to (h, w)
                label = label.astype(np.uint8)
                label = label.reshape(label.shape[0], label.shape[1])

                if img.shape[0:2] == (self.window, self.window) and label.shape[0:2] == (self.window, self.window):
                    self.images.append(img)
                    self.labels.append(label)
                else:
                    # print('wrong shape')
                    pass

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        aug = trfm(image=image, mask=label)
        image = aug['image']
        label = aug['mask']
        return self.as_tensor(image), torch.from_numpy(label)

    def __len__(self):
        return self.len

    def get_np(self, index):
        image, label = self.images[index], self.labels[index]
        return image, label

    def plot(self, index):
        image, label = self.images[index], self.labels[index]
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(image)
        ax = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(label)
        pass

def train(name, model, dataset):
    experiment = Experiment(
        api_key="Dldgid8Kqsrey6mvx8IcTAZD1",
        project_name="deeplearning_UAV_formal",
        workspace="gentry",
    )

    experiment.set_name(name)
    experiment.log_parameters(parameters)

    best_loss = 10
    step = 0

    valid_idx, train_idx = [], []
    for i in range(len(dataset)):
        if i % parameters['split'] == 0:
            valid_idx.append(i)
        else:
            train_idx.append(i)

    train_d = D.Subset(dataset, train_idx)
    valid_d = D.Subset(dataset, valid_idx)

    train_loader = D.DataLoader(train_d, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = D.DataLoader(valid_d, batch_size=bs, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-4, weight_decay=1e-3)
    print(header)
    for epoch in range(1, epochs + 1):
        model.to(DEVICE)
        # print('epoch:{}'.format(epoch))
        losses = []
        start_time = time.time()
        model.train()
        for image, target in train_loader:

            image, target = image.to(DEVICE), target.long().to(DEVICE)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        step += 1
        experiment.log_metric('loss', loss.item(), step=step, epoch=epoch)
        vloss = validation(model, val_loader, criterion)

        experiment.log_metric('vloss', vloss.item(), step=step, epoch=epoch)
        print(raw_line.format(epoch, np.array(losses).mean(), vloss,
                              (time.time() - start_time) / 60 ** 1))
        eval_list = evalution_validation(model, val_loader)
        experiment.log_metrics(eval_list, step=step, epoch=epoch)

        if vloss < best_loss:
            best_loss = vloss
            print('model saving......')
            torch.save(model.state_dict(), model_path.format(name))

    Experiment.end(experiment)
    pass

if __name__ == '__main__':
    print(torch.__version__)
    # set the hyper_parameters
    parameters = {'batch_size':bs,
                  'epochs':epochs,
                  'split':5,
                  'img_size':img_size,
                  'overlap':overlap,
                  'loss_function':'Cross_Entropy',
                  'optimizatiion':'Adam_w',
                  'lr' :1e-4,
                  'weight_decay':1e-3}

    veg = VegDataset(256, 128)

    # Unet_b = smp.Unet(
    #     encoder_name="efficientnet-b7",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=n_classes,  # model output channels (number of classes in your dataset)
    # )

    DeepLab = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet',
                                encoder_output_stride=16, decoder_channels=256, decoder_atrous_rates=(12, 24, 36),
                                in_channels=3, classes=n_classes, activation=None, upsampling=4, aux_params=None)

    t = '2022_name'
    train("DeepLab{}".format(t),DeepLab, veg)
    print('All Finished!')




