import sys
import os
from os.path import join, expanduser

# ADD pytorch_mask_rcnn to sys path
sys.path.append('./pytorch-mask-rcnn')

import pandas as pd
import numpy as np
import imageio
import skimage
import torch
import model as modellib

from dsb_config import DsbConfig, InferenceConfig
from dsb_utils import train_valid_split, split_on_column
from dsb_dataset import DsbDataset

ROOT_DIR = os.getcwd()
HOME_DIR = expanduser('~')
DSB_DATA_DIR = join(HOME_DIR, '.kaggle/competitions/data-science-bowl-2018/')
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

if __name__ == '__main__':
    dsb_config = DsbConfig()
    inference_config = InferenceConfig()

    meta = pd.read_csv(join(DSB_DATA_DIR, 'stage1_metadata.csv'))
    meta_ts = meta[meta['is_train']==0]
    meta_train, meta_valid = train_valid_split( meta[meta['is_train']==1],0.2,[0])

    dsb_dir = join(DSB_DATA_DIR, 'stage1_train')
    train_ids = meta_train.ImageId.values
    val_ids = meta_valid.ImageId.values
    test_dir = join(DSB_DATA_DIR, 'stage1_test')
    test_ids = os.listdir(test_dir)

    # Training dataset
    dataset_train = DsbDataset()
    dataset_train.load_dataset(train_ids, dsb_dir)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DsbDataset()
    dataset_val.load_dataset(val_ids, dsb_dir)
    dataset_val.prepare()

    # # Test dataset
    # dataset_test = DsbDataset()
    # dataset_test.load_dataset(test_ids, test_dir, train_mode=False)
    # dataset_test.prepare()

    model = modellib.MaskRCNN(config=dsb_config, model_dir='./logs')
    model = model.cuda()

    state_dict = torch.load(COCO_MODEL_PATH)
    # class of COCO is 81 while dsb is 2
    # so we should remove the corresponding parameters
    state_dict.pop("mask.conv5.bias")
    state_dict.pop("mask.conv5.weight")
    state_dict.pop("classifier.linear_class.bias")
    state_dict.pop("classifier.linear_class.weight")
    state_dict.pop("classifier.linear_bbox.bias")
    state_dict.pop("classifier.linear_bbox.weight")

    model.load_state_dict(state_dict, strict=False)
    model.train_model(dataset_train, dataset_val,
        learning_rate=dsb_config.LEARNING_RATE,
        epochs=40,
        layers='all')
