#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import os
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import torchvision
import random

from losses.multiscaleloss import *
from time import time

from core.test import test
from models.VGG19 import VGG19


def train(cfg, init_epoch, dataset_loader, train_transforms, val_transforms,
                                  deblurnet, deblurnet_solver, deblurnet_lr_scheduler,
                                  ckpt_dir, train_writer, val_writer,
                                  Best_Img_PSNR, Best_Epoch):


    n_itr = 0
    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Set up data loader
        train_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN, train_transforms),
            batch_size=cfg.CONST.TRAIN_BATCH_SIZE,
            num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=True)

        # Tick / tock
        epoch_start_time = time()
        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        deblur_mse_losses = utils.network_utils.AverageMeter()
        if cfg.TRAIN.USE_PERCET_LOSS == True:
            deblur_percept_losses = utils.network_utils.AverageMeter()
        deblur_losses = utils.network_utils.AverageMeter()
        img_PSNRs = utils.network_utils.AverageMeter()

        # Adjust learning rate
        deblurnet_lr_scheduler.step()
        print('[INFO] learning rate: {0}\n'.format(deblurnet_lr_scheduler.get_lr()))

        batch_end_time = time()
        seq_num = len(train_data_loader)

        vggnet = VGG19()
        if torch.cuda.is_available():
            vggnet = torch.nn.DataParallel(vggnet).cuda()

        for seq_idx, (_, seq_blur, seq_clear) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)
            # Get data from data loader
            seq_blur  = [utils.network_utils.var_or_cuda(img) for img in seq_blur]
            seq_clear = [utils.network_utils.var_or_cuda(img) for img in seq_clear]

            # switch models to training mode
            deblurnet.train()

            # Train the model
            last_img_blur = seq_blur[0]
            output_last_img = seq_blur[0]
            output_last_fea = None
            for batch_idx, [img_blur, img_clear] in enumerate(zip(seq_blur, seq_clear)):
                img_blur_hold = img_blur
                output_img, output_fea = deblurnet(img_blur, last_img_blur, output_last_img, output_last_fea)

                # deblur loss
                deblur_mse_loss = mseLoss(output_img, img_clear)
                deblur_mse_losses.update(deblur_mse_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
                if cfg.TRAIN.USE_PERCET_LOSS == True:
                    deblur_percept_loss = perceptualLoss(output_img, img_clear, vggnet)
                    deblur_percept_losses.update(deblur_percept_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
                    deblur_loss = deblur_mse_loss + 0.01 * deblur_percept_loss
                else:
                    deblur_loss = deblur_mse_loss
                deblur_losses.update(deblur_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
                img_PSNR = PSNR(output_img, img_clear)
                img_PSNRs.update(img_PSNR.item(), cfg.CONST.TRAIN_BATCH_SIZE)

                # deblurnet update
                deblurnet_solver.zero_grad()
                deblur_loss.backward()
                deblurnet_solver.step()

                # Append loss to TensorBoard
                train_writer.add_scalar('STFANet/DeblurLoss_0_TRAIN', deblur_loss.item(), n_itr)
                train_writer.add_scalar('STFANet/DeblurMSELoss_0_TRAIN', deblur_mse_loss.item(), n_itr)
                if cfg.TRAIN.USE_PERCET_LOSS == True:
                    train_writer.add_scalar('STFANet/DeblurPerceptLoss_0_TRAIN', deblur_percept_loss.item(), n_itr)
                n_itr = n_itr + 1

                # Tick / tock
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()

                # print per batch
                if (batch_idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                    if cfg.TRAIN.USE_PERCET_LOSS == True:
                        print('[TRAIN] [Ech {0}/{1}][Seq {2}/{3}][Bch {4}/{5}] BT {6} DT {7} DeblurLoss {8} [{9}, {10}] PSNR {11}'
                            .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, seq_idx + 1, seq_num, batch_idx + 1,
                                    cfg.DATA.SEQ_LENGTH, batch_time, data_time,
                                    deblur_losses, deblur_mse_losses, deblur_percept_losses, img_PSNRs))
                    else:
                        print(
                            '[TRAIN] [Ech {0}/{1}][Seq {2}/{3}][Bch {4}/{5}] BT {6} DT {7} DeblurLoss {8} PSNR {9}'
                            .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, seq_idx + 1, seq_num, batch_idx + 1,
                                    cfg.DATA.SEQ_LENGTH, batch_time, data_time, deblur_losses, img_PSNRs))

                # show
                if seq_idx == 0 and batch_idx < cfg.TEST.VISUALIZATION_NUM:
                    img_blur = img_blur[0][[2, 1, 0], :, :].cpu() + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    img_clear = img_clear[0][[2, 1, 0], :, :].cpu() + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    output_last_img = output_last_img[0][[2, 1, 0], :, :].cpu() + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    img_out = output_img[0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)

                    result = torch.cat([torch.cat([img_blur, img_clear], 2),
                                        torch.cat([output_last_img, img_out], 2)], 1)
                    result = torchvision.utils.make_grid(result, nrow=1, normalize=True)
                    train_writer.add_image('STFANet/TRAIN_RESULT' + str(batch_idx + 1), result, epoch_idx + 1)

                # *** Update output_last_img/feature ***
                last_img_blur = img_blur_hold
                output_last_img = output_img.clamp(0.0, 1.0).detach()
                output_last_fea = output_fea.detach()

            # print per sequence
            print('[TRAIN] [Epoch {0}/{1}] [Seq {2}/{3}] ImgPSNR_avg {4}\n'
                  .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, seq_idx + 1, seq_num, img_PSNRs.avg))

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('STFANet/EpochPSNR_0_TRAIN', img_PSNRs.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        print('[TRAIN] [Epoch {0}/{1}]\t EpochTime {2}\t ImgPSNR_avg {3}\n'
              .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, img_PSNRs.avg))

        # Validate the training models
        img_PSNR = test(cfg, epoch_idx, dataset_loader, val_transforms, deblurnet, val_writer)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % (epoch_idx + 1)), \
                                                      epoch_idx + 1, deblurnet, deblurnet_solver, \
                                                      Best_Img_PSNR, Best_Epoch)
        if img_PSNR >= Best_Img_PSNR:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            Best_Img_PSNR = img_PSNR
            Best_Epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'best-ckpt.pth.tar'), \
                                                      epoch_idx + 1, deblurnet, deblurnet_solver, \
                                                      Best_Img_PSNR, Best_Epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()