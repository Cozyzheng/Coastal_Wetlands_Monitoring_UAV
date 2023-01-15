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

import numpy as np
import six
import torch 
from cfg import *
# from train import *
# from predict import *

def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    # pred.shape(h, w, 1)
    pred_labels = iter(pred_labels)   
    gt_labels = iter(gt_labels)   
    n_class = n_classes
    confusion = np.zeros((n_class, n_class), dtype=np.int64)   
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        # print(pred_label.shape, gt_label.shape)
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()  
        gt_label = gt_label.flatten()  

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        # print(lb_max)
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels. 
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) + pred_label[mask],
            minlength=n_class ** 2)\
            .reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    return confusion


def calc_semantic_segmentation_iou(confusion):
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                       - np.diag(confusion))
    iou = np.diag(confusion) / (iou_denominator+1e-10)
    return iou
    # return iou


def eval_semantic_segmentation(pred_labels, gt_labels):
    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels)

    iou = calc_semantic_segmentation_iou(confusion)     # (1２, )

    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()

    class_accuracy = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)  # (1２, )

    return {'iou': iou, 'miou': np.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy)}
            # 'mean_class_accuracy': np.nanmean(class_accuracy)}

def Precision(con):
    # sum
    pre = np.diag(con).sum()/(con.sum(axis=0) + 1e-10)
    return pre

def Recall(con):
    re = np.diag(con).sum()/(con.sum(axis=1)+ 1e-10)
    return re

def F1_Score(con):
    pre = Precision(con)
    re = Recall(con)
    F1_score = 2 * pre * re /(pre + re + 1e-10)
    return  F1_score


@torch.no_grad()
def evalution_validation(model, loader):
    miou = []
    pa = []
    re = []
    Fscore = []
    mean_class_accuracy = []

    class_0_acc = []
    class_1_acc = []
    class_2_acc = []
    class_3_acc = []
    class_4_acc = []

    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.long().to(DEVICE)
        output = model(image).cpu().numpy()

        # cup, numpy,argmax
        output = np.argmax(output, axis=1)
        target = target.cpu().numpy()
        output = output.astype(np.uint8)
        target = target.astype(np.uint8)

        # cal
        confusion = calc_semantic_segmentation_confusion(
            output, target)

        iou = calc_semantic_segmentation_iou(confusion)

        recall = Recall(confusion)

        F1 = F1_Score(confusion)

        re.append(recall)

        Fscore.append(F1)

        miou.append(np.mean(iou))

        pa.append(np.diag(confusion).sum() / (confusion.sum() + 1e-10))

        class_accuracy = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)
        class_0_acc.append(class_accuracy[0])
        class_1_acc.append(class_accuracy[1])
        class_2_acc.append(class_accuracy[2])
        class_3_acc.append(class_accuracy[3])

        mean_class_accuracy.append(np.nanmean(class_accuracy))

    # poch
    miou = np.mean(miou)
    pa = np.mean(pa)
    re = np.mean(re)
    F1_score = np.mean(Fscore)
    class_0 = np.nanmean(class_0_acc)
    class_1 = np.nanmean(class_1_acc)
    class_2 = np.nanmean(class_2_acc)
    class_3 = np.nanmean(class_3_acc)


    return {'miou': np.mean(miou), 'pixel_accuracy': np.mean(pa), 'recall':re,
            'f1-score':F1_score,
            'mean_class_accuracy': np.mean(mean_class_accuracy),
            'class_0_acc':class_0,
            'class_1_acc':class_1,
            'class_2_acc':class_2,
            'class_3_acc':class_3}

@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.long().to(DEVICE)
        output = model(image)
        loss = loss_fn(output, target)
        losses.append(loss.item())

    return np.array(losses).mean()

if __name__ == '__main__':
    Unet = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=5,  # model output channels (number of classes in your dataset)
    )
