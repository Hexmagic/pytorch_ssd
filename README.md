[中文版本](cn_README.md)

# pytorch_ssd
simple implementation of ssd by pytorch

## Data

download voc data from network

    wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
    wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
    wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    tar xf VOCtrainval_11-May-2012.tar
    tar xf VOCtrainval_06-Nov-2007.tar
    tar xf VOCtest_06-Nov-2007.tar

rename VOCdevkit to datasets:

    D:\PYTORCH_SSD\DATASETS
    ├─VOC2007
    │  ├─Annotations
    │  ├─ImageSets
    │  │  ├─Layout
    │  │  ├─Main
    │  │  └─Segmentation
    │  ├─JPEGImages
    │  ├─labels
    │  ├─SegmentationClass
    │  └─SegmentationObject
    └─VOC2012
        ├─Annotations
        ├─ImageSets
        │  ├─Action
        │  ├─Layout
        │  ├─Main
        │  └─Segmentation
        ├─JPEGImages
        ├─labels
        ├─SegmentationClass
        └─SegmentationObject

## Train

    python train.py --data_dir datasets

## Pretrained Model Download

[Google Drive](https://drive.google.com/file/d/10-ps44l6uIgjRjCzFiRXF3W6Cyyy2E2E/view?usp=sharing)

## Test

    python inference.py --pretrained_weight 80000_ssd300.pth

this is the output:


| Field 1     | Field 2 |
| ----------- | ------- |
| 类别        | 值      |
| mAP         | 0.71    |
| aeroplane   | 0.82    |
| bicycle     | 0.8     |
| bird        | 0.7     |
| boat        | 0.6     |
| bottle      | 0.45    |
| bus         | 0.79    |
| car         | 0.78    |
| cat         | 0.85    |
| chair       | 0.54    |
| cow         | 0.66    |
| diningtable | 0.61    |
| dog         | 0.8     |
| horse       | 0.78    |
| motorbike   | 0.82    |
| person      | 0.81    |
| pottedplant | 0.46    |
| sheep       | 0.71    |
| sofa        | 0.69    |
| train       | 0.84    |
| tvmonitor   | 0.68    |

