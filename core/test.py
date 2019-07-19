#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from losses.multiscaleloss import *
import torchvision

import numpy as np
import scipy.io as io


from time import time

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)

def test(cfg, epoch_idx, dataset_loader, test_transforms, deblurnet, test_writer):
    # Set up data loader
    test_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST, test_transforms),
        batch_size=cfg.CONST.TEST_BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)

    seq_num = len(test_data_loader)
    # Batch average meterics
    batch_time = utils.network_utils.AverageMeter()
    test_time = utils.network_utils.AverageMeter()
    data_time = utils.network_utils.AverageMeter()
    img_PSNRs = utils.network_utils.AverageMeter()
    batch_end_time = time()
    test_psnr = dict()
    g_names= 'init'

    for seq_idx, (name, seq_blur, seq_clear) in enumerate(test_data_loader):
        data_time.update(time() - batch_end_time)

        seq_blur = [utils.network_utils.var_or_cuda(img) for img in seq_blur]
        seq_clear = [utils.network_utils.var_or_cuda(img) for img in seq_clear]
        seq_len = len(seq_blur)
        # Switch models to training mode
        deblurnet.eval()

        if cfg.NETWORK.PHASE == 'test':
            if not g_names == name[0]:
                g_names = name[0]
                save_num = 0

            assert (len(name) == 1)
            name = name[0]
            if not name in test_psnr:
                test_psnr[name] = {
                    'n_samples': 0,
                    'psnr': []
                }
        with torch.no_grad():
            last_img_blur = seq_blur[0]
            output_last_img = seq_blur[0]
            output_last_fea = None
            for batch_idx, [img_blur, img_clear] in enumerate(zip(seq_blur, seq_clear)):
                img_blur_hold = img_blur
                # Test runtime
                torch.cuda.synchronize()
                test_time_start = time()
                # --Forwards--
                output_img, output_fea = deblurnet(img_blur, last_img_blur, output_last_img, output_last_fea)
                torch.cuda.synchronize()
                test_time.update(time() - test_time_start)
                print('[RUNING TIME] {0}'.format(test_time))

                img_PSNR = PSNR(output_img, img_clear)
                img_PSNRs.update(img_PSNR.item(), cfg.CONST.TRAIN_BATCH_SIZE)

                batch_time.update(time() - batch_end_time)
                batch_end_time = time()

                # Print per batch
                if (batch_idx+1) % cfg.TEST.PRINT_FREQ == 0:
                    print('[TEST] [Ech {0}/{1}][Seq {2}/{3}][Bch {4}/{5}] BT {6} DT {7}\t imgPSNR {8}'
                          .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, seq_idx+1, seq_num, batch_idx + 1, seq_len, batch_time, data_time, img_PSNRs))

                if seq_idx == 0 and batch_idx < cfg.TEST.VISUALIZATION_NUM and not cfg.NETWORK.PHASE == 'test':

                    if epoch_idx == 0 or cfg.NETWORK.PHASE in ['test','resume']:
                        img_blur = img_blur[0][[2, 1, 0], :, :].cpu() + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                        img_clear = img_clear[0][[2, 1, 0], :, :].cpu() + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                        test_writer.add_image('STFANet/IMG_BLUR' + str(batch_idx + 1), img_blur, epoch_idx + 1)
                        test_writer.add_image('STFANet/IMG_CLEAR' + str(batch_idx + 1), img_clear, epoch_idx + 1)

                        output_last_img = output_last_img[0][[2, 1, 0], :, :].cpu() + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)
                    img_out = output_img[0][[2, 1, 0], :, :].cpu().clamp(0.0, 1.0) + torch.Tensor(cfg.DATA.MEAN).view(3, 1, 1)

                    test_writer.add_image('STFANet/LAST_IMG_OUT' +str(batch_idx + 1), output_last_img, epoch_idx + 1)
                    test_writer.add_image('STFANet/IMAGE_OUT' +str(batch_idx + 1), img_out, epoch_idx + 1)

                if cfg.NETWORK.PHASE == 'test':
                    test_psnr[name]['n_samples'] += 1
                    test_psnr[name]['psnr'].append(img_PSNR)
                    img_dir = os.path.join(cfg.DIR.OUT_PATH, name)
                    if not os.path.isdir(img_dir):
                        mkdir(img_dir)

                    print('[TEST Saving: ]'+img_dir + '/' + str(save_num).zfill(5) + '.png')

                    cv2.imwrite(img_dir + '/' + str(save_num).zfill(5) + '.png',
                                (output_img.clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
                                    np.uint8),[int(cv2.IMWRITE_PNG_COMPRESSION), 5])

                    save_num = save_num + 1

                # *** Update output_last_img/feature ***
                last_img_blur = img_blur_hold
                output_last_img = output_img.clamp(0.0, 1.0)
                output_last_fea = output_fea

            # Print per seq
            if (batch_idx + 1) % cfg.TEST.PRINT_FREQ == 0:
                print(
                    '[TEST] [Ech {0}/{1}][Seq {2}/{3}] BT {4} DT {5} \t ImgPSNR_avg {6}\n'
                    .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, seq_idx + 1, seq_num, batch_time, data_time, img_PSNRs.avg))



    # Output testing results
    if cfg.NETWORK.PHASE == 'test':

        # Output test results
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))
        for name in test_psnr:
            test_psnr[name]['psnr'] = np.mean(test_psnr[name]['psnr'], axis=0)
            print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}'.format(name, test_psnr[name]['n_samples'],
                                                                        test_psnr[name]['psnr']))

        result_file = open(os.path.join(cfg.DIR.OUT_PATH, 'test_result.txt'), 'w')
        sys.stdout = result_file
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))
        for name in test_psnr:
            print('[TEST] Name: {0}\t Num: {1}\t Mean_PSNR: {2}'.format(name, test_psnr[name]['n_samples'],
                                                                        test_psnr[name]['psnr']))
        result_file.close()
    else:
        # Output val results
        print('============================ TEST RESULTS ============================')
        print('[TEST] Total_Mean_PSNR:' + str(img_PSNRs.avg))
        print('[TEST] [Epoch{0}]\t BatchTime_avg {1}\t DataTime_avg {2}\t ImgPSNR_avg {3}\n'
              .format(cfg.TRAIN.NUM_EPOCHES, batch_time.avg, data_time.avg, img_PSNRs.avg))

        # Add testing results to TensorBoard
        test_writer.add_scalar('STFANet/EpochPSNR_1_TEST', img_PSNRs.avg, epoch_idx + 1)

        return img_PSNRs.avg