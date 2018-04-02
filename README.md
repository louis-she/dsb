# Mask RCNN in pytorch solution for Kaggle's 2018 Data Science Bowl

This is a solution of 2018 Data Science Bowl rely on a forked version of multimodallearning's [pytorch-mask-rcnn](https://github.com/multimodallearning/pytorch-mask-rcnn).

## Requirements

* Python 3.6
* Pytorch 0.3
* 8GB memory or better GPU

## How to use

1. Clone this repo.
```
git clone https://github.com/louis-she/dsb.git --recursive
```

2. Download data by kaggle command.
```
kaggle competitions download -c data-science-bowl-2018
```

3. Two more repositories that need to be build with the right `--arch` option for cuda support.
The two functions are Non-Maximum Suppression from ruotianluo's [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)
repository and longcw's [RoiAlign](https://github.com/longcw/RoIAlign.pytorch).

| GPU | arch |
| --- | --- |
| TitanX | sm_52 |
| GTX 960M | sm_50 |
| GTX 1070 | sm_61 |
| GTX 1080 (Ti) | sm_61 |

```
cd nms/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
cd ../../
python build.py
cp -R _ext dsb/pytorch-mask-rcnn/nms/_ext

cd roialign/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
cd ../../
python build.py
cp -R _ext dsb/pytorch-mask-rcnn/roialign/roi_align/_ext
```

4. Download pretrained weights on COCO dataset. https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

5. Using the following command to train the model

```
python dsb_train.py
```

6. Tweak the configures in `dsb_config.py`, the learning rate and training strategy can be modified directly in `dsb_train.py`.

## Result

I don't have enough GPU resources to train this. I train this 2 epoch with coco pretrained model and I got 0.421 LB.