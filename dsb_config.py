from os import getcwd
from os.path import join, expanduser

import numpy as np
from config import Config

class DsbConfig(Config):

    # DSB data root directory
    ROOT_DIR = getcwd()
    HOME_DIR = expanduser('~')
    DSB_DATA_DIR = join(HOME_DIR, '.kaggle/competitions/data-science-bowl-2018/')
    LOG_DIR = join(ROOT_DIR, './logs')

    # Weights path, if one wants to resume from last run, change this
    # options together with the IS_COCO_STATE
    STATE_DICT_PATH = join(ROOT_DIR, "mask_rcnn_coco.pth")

    # If the state dict file is mask_rcnn_coco, set this to True
    # If resume from last run, set this to False
    IS_COCO_STATE = True

    # Give the configuration a recognizable name
    NAME = "dsb"

    # Validation dataset proportion
    VALIDATION_PROPORTION = 0.1

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution image
    # Tweak this
    USE_MINI_MASK = True

    # If the previous is False, then this is not used
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Train on 1 GPU and 8 images per GPU. Batch size is GPUs * images/GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch.
    # typically be equal to the number of samples of your dataset divided by the batch size
    STEPS_PER_EPOCH = 612
    VALIDATION_STEPS = 58

    # Number of classes (including background)
    NUM_CLASSES = 2

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    IMAGE_PADDING = True  # currently, the False option is not supported

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels, maybe add a 256?
    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 320 #300

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 2000
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    TRAIN_ROIS_PER_IMAGE = 512
    RPN_NMS_THRESHOLD = 0.7
    MAX_GT_INSTANCES = 256
    DETECTION_MAX_INSTANCES = 500
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7 # may be smaller?
    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3 # 0.3

    MEAN_PIXEL = np.array([0.,0.,0.])

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

class InferenceConfig(DsbConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # MEAN_PIXEL = np.array([56.02288505, 54.02376286, 54.26675248])
