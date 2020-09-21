#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

from easydict import EasyDict as edict
import os
import socket

__C     = edict()
cfg     = __C

#
# Common
#
__C.CONST                               = edict()
__C.CONST.DEVICE                        = 'all'                   # '0'
__C.CONST.NUM_WORKER                    = 1                       # number of data workers
__C.CONST.WEIGHTS                       = './ckpt/best-ckpt.pth.tar'
__C.CONST.TRAIN_BATCH_SIZE              = 1
__C.CONST.TEST_BATCH_SIZE               = 1
#
# Dataset
#
__C.DATASET                             = edict()
__C.DATASET.DATASET_NAME                = 'VideoDeblur'       # VideoDeblur, VideoDeblurReal

#
# Directories
#
__C.DIR                                 = edict()
__C.DIR.OUT_PATH = './output'

if cfg.DATASET.DATASET_NAME == 'VideoDeblur':
    __C.DIR.DATASET_JSON_FILE_PATH = './datasets/VideoDeblur.json'
    __C.DIR.DATASET_ROOT = '/data/DeepVideoDeblurringDataset/'
    __C.DIR.IMAGE_BLUR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/%s/input/%s.jpg')
    __C.DIR.IMAGE_CLEAR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/%s/GT/%s.jpg')
# real
elif cfg.DATASET.DATASET_NAME == 'VideoDeblurReal':
    __C.DIR.DATASET_JSON_FILE_PATH = './datasets/VideoDeblurReal.json'
    __C.DIR.DATASET_ROOT = '/data/qualitative_datasets/'
    __C.DIR.IMAGE_BLUR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/%s/input/%s.jpg')
    __C.DIR.IMAGE_CLEAR_PATH = os.path.join(__C.DIR.DATASET_ROOT,'%s/%s/input/%s.jpg')

#
# data augmentation
#
__C.DATA                                = edict()
__C.DATA.STD                            = [255.0, 255.0, 255.0]
__C.DATA.MEAN                           = [0.0, 0.0, 0.0]
__C.DATA.CROP_IMG_SIZE                  = [320, 448]              # Crop image size: height, width
__C.DATA.GAUSSIAN                       = [0, 1e-4]               # mu, std_var
__C.DATA.COLOR_JITTER                   = [0.2, 0.15, 0.3, 0.1]   # brightness, contrast, saturation, hue
__C.DATA.SEQ_LENGTH                     = 20


#
# Network
#
__C.NETWORK                             = edict()
__C.NETWORK.DEBLURNETARCH               = 'DeblurNet'             # available options: DeblurNet
__C.NETWORK.LEAKY_VALUE                 = 0.1
__C.NETWORK.BATCHNORM                   = False
__C.NETWORK.PHASE                       = 'test'                 # available options: 'train', 'test', 'resume'


#
# Training
#

__C.TRAIN                               = edict()
__C.TRAIN.USE_PERCET_LOSS               = True
__C.TRAIN.NUM_EPOCHES                   = 400                    # maximum number of epoches
__C.TRAIN.LEARNING_RATE                 = 1e-4
__C.TRAIN.LR_MILESTONES                 = [80,160,250]
__C.TRAIN.LR_DECAY                      = 0.1                    # Multiplicative factor of learning rate decay
__C.TRAIN.MOMENTUM                      = 0.9
__C.TRAIN.BETA                          = 0.999
__C.TRAIN.BIAS_DECAY                    = 0.0                    # regularization of bias, default: 0
__C.TRAIN.WEIGHT_DECAY                  = 0.0                    # regularization of weight, default: 0
__C.TRAIN.PRINT_FREQ                    = 10
__C.TRAIN.SAVE_FREQ                     = 10                     # weights will be overwritten every save_freq epoch

__C.LOSS                                = edict()
__C.LOSS.MULTISCALE_WEIGHTS             = [0.3, 0.3, 0.2, 0.1, 0.1]

#
# Testing options
#
__C.TEST                                = edict()
__C.TEST.VISUALIZATION_NUM              = 10
__C.TEST.PRINT_FREQ                     = 5
if __C.NETWORK.PHASE == 'test':
    __C.CONST.TEST_BATCH_SIZE           = 1
