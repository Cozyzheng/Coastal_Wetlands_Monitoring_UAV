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
import RSjunyi.rs as grs
import rasterio as rs
import torch

imgs_path = '/media/gentry/new_label_2/img'
labels_path = '/media/gentry/new_label_2/label'
model_path = '/media/gentry/model_pth/{}.pth'
predict_path ='/media/gentry/predict/{}_{}.tif'
pre1_path = '/media/gentry/1.tif'
pre7_path = '/media/gentry/7.tif'

# class
n_classes = 5

# batch_size
bs =32

img_size = 256

overlap = 64

epochs = 1000

best_loss = 10

### Table for results
header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time, m
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}'*2 + '\u2502{:6.2f}'


def geo_trans(path_geo, path_tar):
    '''
    :param path_geo:
    :param path_tar:
    :return:
    '''
    # Check the shape[:1]
    if grs.img_read(path_geo, info=False).shape[:2] != grs.img_read(path_tar, info=False).shape[:2]:
        raise ValueError('invalid shape')
    img_geo = rs.open(path_geo)
    if img_geo.crs != None:
        img = rs.open(path_tar, mode='r+')
        img.crs = img_geo.crs
        img.transform = img_geo.transform
        # img.closed()
        print('GeoInfo Trans: GeoInfo Added')
    else:
        print('GeoInfo Trans: Deficiency in GeoInfo of Input file')
    pass
