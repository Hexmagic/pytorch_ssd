# pytorch_ssd

### 下载数据集

    wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
    wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
    wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    tar xf VOCtrainval_11-May-2012.tar
    tar xf VOCtrainval_06-Nov-2007.tar
    tar xf VOCtest_06-Nov-2007.tar

得到VOCdevkit文件夹，重命名为datasets,如下

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

## 训练

    python train.py --data_dir datasets