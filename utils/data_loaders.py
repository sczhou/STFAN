#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import cv2
import json
import numpy as np
import os
import io
import random
import scipy.io
import sys
import torch.utils.data.dataset

from config import cfg
from datetime import datetime as dt
from enum import Enum, unique
from utils.imgio_gen import readgen
import utils.network_utils

class DatasetType(Enum):
    TRAIN = 0
    TEST  = 1


class VideoDeblurDataset(torch.utils.data.dataset.Dataset):
    """VideoDeblurDataset class used for PyTorch DataLoader"""

    def __init__(self, file_list_with_metadata, transforms = None):
        self.file_list = file_list_with_metadata
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name, seq_blur, seq_clear = self.get_datum(idx)
        seq_blur, seq_clear = self.transforms(seq_blur, seq_clear)
        return name, seq_blur, seq_clear

    def get_datum(self, idx):

        name = self.file_list[idx]['name']
        length = self.file_list[idx]['length']
        seq_blur_paths = self.file_list[idx]['seq_blur']
        seq_clear_paths = self.file_list[idx]['seq_clear']
        seq_blur = []
        seq_clear = []
        for i in range(length):
            img_blur = readgen(seq_blur_paths[i]).astype(np.float32)
            img_clear = readgen(seq_clear_paths[i]).astype(np.float32)
            seq_blur.append(img_blur)
            seq_clear.append(img_clear)

        return name, seq_blur, seq_clear
# //////////////////////////////// = End of VideoDeblurDataset Class Definition = ///////////////////////////////// #

class VideoDeblurDataLoader:
    def __init__(self):
        self.img_blur_path_template = cfg.DIR.IMAGE_BLUR_PATH
        self.img_clear_path_template = cfg.DIR.IMAGE_CLEAR_PATH

        # Load all files of the dataset
        with io.open(cfg.DIR.DATASET_JSON_FILE_PATH, encoding='utf-8') as file:
            self.files_list = json.loads(file.read())

    def get_dataset(self, dataset_type, transforms=None):
        sequences = []
        # Load data for each sequence
        for file in self.files_list:
            if dataset_type == DatasetType.TRAIN and file['phase'] == 'train':
                name = file['name']
                phase = file['phase']
                samples = file['sample']
                sam_len = len(samples)
                seq_len = cfg.DATA.SEQ_LENGTH
                seq_num = int(sam_len/seq_len)
                for n in range(seq_num):
                    sequence = self.get_files_of_taxonomy(phase, name, samples[seq_len*n: seq_len*(n+1)])
                    sequences.extend(sequence)

                if not seq_len%seq_len == 0:
                    sequence = self.get_files_of_taxonomy(phase, name, samples[-seq_len:])
                    sequences.extend(sequence)
                    seq_num += 1

                print('[INFO] %s Collecting files of Taxonomy [Name = %s]' % (dt.now(), name + ': ' + str(seq_num)))


            elif dataset_type == DatasetType.TEST and file['phase'] == 'test':
                name = file['name']
                phase = file['phase']
                samples = file['sample']
                sam_len = len(samples)
                seq_len = cfg.DATA.SEQ_LENGTH
                seq_num = int(sam_len / seq_len)
                for n in range(seq_num):
                    sequence = self.get_files_of_taxonomy(phase, name, samples[seq_len*n: seq_len*(n+1)])
                    sequences.extend(sequence)

                if not seq_len % seq_len == 0:
                    sequence = self.get_files_of_taxonomy(phase, name, samples[-seq_len:])
                    sequences.extend(sequence)
                    seq_num += 1

                print('[INFO] %s Collecting files of Taxonomy [Name = %s]' % (dt.now(), name + ': ' + str(seq_num)))

        print('[INFO] %s Complete collecting files of the dataset for %s. Seq Numbur: %d.\n' % (dt.now(), dataset_type.name, len(sequences)))
        return VideoDeblurDataset(sequences, transforms)

    def get_files_of_taxonomy(self, phase, name, samples):
        n_samples = len(samples)
        seq_blur_paths = []
        seq_clear_paths = []
        sequence = []

        for sample_idx, sample_name in enumerate(samples):
            # Get file path of img
            img_blur_path = self.img_blur_path_template % (phase, name, sample_name)
            img_clear_path = self.img_clear_path_template % (phase, name, sample_name)
            if os.path.exists(img_blur_path) and os.path.exists(img_clear_path):
                seq_blur_paths.append(img_blur_path)
                seq_clear_paths.append(img_clear_path)

        if not seq_blur_paths == [] and not seq_clear_paths == []:
            if phase == 'train' and random.random() < 0.5:
                # reverse
                seq_blur_paths.reverse()
                seq_clear_paths.reverse()
            sequence.append({
                'name': name,
                'length': n_samples,
                'seq_blur': seq_blur_paths,
                'seq_clear': seq_clear_paths,
            })
        return sequence
# /////////////////////////////// = End of VideoDeblurDataLoader Class Definition = /////////////////////////////// #


DATASET_LOADER_MAPPING = {
    'VideoDeblur': VideoDeblurDataLoader,
    'VideoDeblurReal': VideoDeblurDataLoader
}
