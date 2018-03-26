# Mask RCNN for kaggle dsb competition

## Requirements
* Python 3.6
* Pytorch 0.3

### How to use

1. Clone this repo.
```
git clone https://github.com/louis-she/dsb.git
```

2. Download data by kaggle command.
```
kaggle competitions download -c data-science-bowl-2018
```

3. We use functions from two more repositories that need to be build with the right `--arch` option for cuda support.
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
cd ../

cd roialign/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
cd ../../
python build.py
cd ../../
```

4. Using the following command to train the model

```
python dsb_train.py
```