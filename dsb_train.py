import sys
import random
from os import listdir, mkdir
from os.path import join, expanduser, isdir

# ADD pytorch_mask_rcnn to sys path
sys.path.append('./pytorch-mask-rcnn')

import pandas as pd
import numpy as np
import imageio
import skimage
import torch
import model as modellib

from dsb_config import DsbConfig, InferenceConfig
from dsb_dataset import DsbDataset

if __name__ == '__main__':
    dsb_config = DsbConfig()
    inference_config = InferenceConfig()

    image_ids = listdir(join(dsb_config.DSB_DATA_DIR, 'stage1_train'))
    random.shuffle(image_ids)
    pivot = int(len(image_ids) * dsb_config.VALIDATION_PROPORTION)
    valid_ids, train_ids = (image_ids[:pivot], image_ids[pivot:])

    # Training dataset
    dataset_train = DsbDataset()
    dataset_train.load_dataset(train_ids, join(dsb_config.DSB_DATA_DIR, 'stage1_train'))
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DsbDataset()
    dataset_val.load_dataset(valid_ids, join(dsb_config.DSB_DATA_DIR, 'stage1_train'))
    dataset_val.prepare()

    if not isdir(dsb_config.LOG_DIR):
        mkdir(dsb_config.LOG_DIR)

    model = modellib.MaskRCNN(config=dsb_config, model_dir=dsb_config.LOG_DIR)
    model = model.cuda()

    state_dict = torch.load(dsb_config.STATE_DICT_PATH)
    # class of COCO is 81 while dsb is 2
    # so we should remove the corresponding parameters

    def state_modifier(state_dict):
        if dsb_config.IS_COCO_STATE:
            state_dict.pop("mask.conv5.bias")
            state_dict.pop("mask.conv5.weight")
            state_dict.pop("classifier.linear_class.bias")
            state_dict.pop("classifier.linear_class.weight")
            state_dict.pop("classifier.linear_bbox.bias")
            state_dict.pop("classifier.linear_bbox.weight")
        return state_dict

    model.load_weights(dsb_config.STATE_DICT_PATH, state_modifier)

    # Tweak the training strategy
    model.train_model(dataset_train, dataset_val,
        learning_rate=0.001,
        epochs=20,
        layers='heads')

    model.train_model(dataset_train, dataset_val,
        learning_rate=0.0001,
        epochs=40,
        layers='all')