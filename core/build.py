#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import os
import sys
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import models
from models.DeblurNet import DeblurNet

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from core.train import train
from core.test import test

from losses.multiscaleloss import *

def bulid_net(cfg):

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark  = True

    # Set up data augmentation
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.ColorJitter(cfg.DATA.COLOR_JITTER),
        utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        utils.data_transforms.RandomCrop(cfg.DATA.CROP_IMG_SIZE),
        utils.data_transforms.RandomVerticalFlip(),
        utils.data_transforms.RandomHorizontalFlip(),
        utils.data_transforms.RandomColorChannel(),
        utils.data_transforms.RandomGaussianNoise(cfg.DATA.GAUSSIAN),
        utils.data_transforms.ToTensor(),
    ])

    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME]()

    # Set up networks
    deblurnet = models.__dict__[cfg.NETWORK.DEBLURNETARCH].__dict__[cfg.NETWORK.DEBLURNETARCH]()

    print('[DEBUG] %s Parameters in %s: %d.' % (dt.now(), cfg.NETWORK.DEBLURNETARCH,
                                                utils.network_utils.count_parameters(deblurnet)))

    # Initialize weights of networks
    deblurnet.apply(utils.network_utils.init_weights_xavier)

    # Set up solver
    a =  filter(lambda p: p.requires_grad, deblurnet.parameters())
    deblurnet_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, deblurnet.parameters()), lr=cfg.TRAIN.LEARNING_RATE,
                                         betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))

    if torch.cuda.is_available():
        deblurnet = torch.nn.DataParallel(deblurnet).cuda()

    # Load pretrained model if exists
    init_epoch       = 0
    Best_Epoch       = -1
    Best_Img_PSNR    = 0


    if cfg.NETWORK.PHASE in ['test','resume']:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        deblurnet.load_state_dict(checkpoint['deblurnet_state_dict'])
        # deblurnet_solver.load_state_dict(checkpoint['deblurnet_solver_state_dict'])
        init_epoch = checkpoint['epoch_idx']+1
        Best_Img_PSNR = checkpoint['Best_Img_PSNR']
        Best_Epoch = checkpoint['Best_Epoch']
        print('[INFO] {0} Recover complete. Current epoch #{1}, Best_Img_PSNR = {2} at epoch #{3}.' \
              .format(dt.now(), init_epoch, Best_Img_PSNR, Best_Epoch))



    # Set up learning rate scheduler to decay learning rates dynamically
    deblurnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(deblurnet_solver,
                                                                   milestones=cfg.TRAIN.LR_MILESTONES,
                                                                   gamma=cfg.TRAIN.LR_DECAY)

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, dt.now().isoformat() + '_' + cfg.NETWORK.DEBLURNETARCH, '%s')
    log_dir      = output_dir % 'logs'
    ckpt_dir     = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    test_writer  = SummaryWriter(os.path.join(log_dir, 'test'))
    print('[INFO] Output_dir： {0}'.format(output_dir[:-2]))

    if cfg.NETWORK.PHASE in ['train','resume']:
        train(cfg, init_epoch, dataset_loader, train_transforms, test_transforms,
                              deblurnet, deblurnet_solver, deblurnet_lr_scheduler,
                              ckpt_dir, train_writer, test_writer,
                              Best_Img_PSNR, Best_Epoch)
    else:
        if os.path.exists(cfg.CONST.WEIGHTS):
            test(cfg, init_epoch, dataset_loader, test_transforms, deblurnet, test_writer)
        else:
            print('[FATAL] %s Please specify the file path of checkpoint.' % (dt.now()))
            sys.exit(2)