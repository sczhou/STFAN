#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import matplotlib
import os
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

# Fix problem: possible deadlock in dataloader
# import cv2
# cv2.setNumThreads(0)

from argparse import ArgumentParser
from pprint import pprint

from config import cfg
from core.build import bulid_net
import torch

def get_args_from_command_line():

    parser = ArgumentParser(description='Parser of Runner of Network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [cuda]', default=cfg.CONST.DEVICE, type=str)
    parser.add_argument('--phase', dest='phase', help='phase of CNN', default=cfg.NETWORK.PHASE, type=str)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=cfg.CONST.WEIGHTS, type=str)
    parser.add_argument('--data', dest='data_path', help='Set dataset root_path', default=cfg.DIR.DATASET_ROOT, type=str)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    args = parser.parse_args()
    return args

def main():

    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if args.phase is not None:
        cfg.NETWORK.PHASE = args.phase
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
    if args.data_path is not None:
        cfg.DIR.DATASET_ROOT = args.data_path
        if cfg.DATASET.DATASET_NAME == 'VideoDeblur':
            cfg.DIR.IMAGE_BLUR_PATH = os.path.join(args.data_path,'%s/%s/input/%s.jpg')
            cfg.DIR.IMAGE_CLEAR_PATH = os.path.join(args.data_path,'%s/%s/GT/%s.jpg')
        if cfg.DATASET.DATASET_NAME == 'VideoDeblurReal':
            cfg.DIR.IMAGE_BLUR_PATH = os.path.join(args.data_path,'%s/%s/input/%s.jpg')
            cfg.DIR.IMAGE_CLEAR_PATH = os.path.join(args.data_path,'%s/%s/input/%s.jpg')
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path


    # Print config
    print('Use config:')
    pprint(cfg)


    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str and not cfg.CONST.DEVICE == 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
    print('CUDA DEVICES NUMBER: '+ str(torch.cuda.device_count()))

    # Setup Network & Start train/test process
    bulid_net(cfg)

if __name__ == '__main__':

    main()
