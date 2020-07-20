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

## 预训练模型下载

[Google Drive](https://drive.google.com/file/d/10-ps44l6uIgjRjCzFiRXF3W6Cyyy2E2E/view?usp=sharing)

## 测试

    python inference.py --pretrained_weight 80000_ssd300.pth

这里是测试结果，训练了80000个epoch，没有最后的微调
```
+-------------+---------+
|   Field 1   | Field 2 |
+-------------+---------+
|     类别    |    值   |
|     mAP     |   0.71  |
|  aeroplane  |   0.82  |
|   bicycle   |   0.8   |
|     bird    |   0.7   |
|     boat    |   0.6   |
|    bottle   |   0.45  |
|     bus     |   0.79  |
|     car     |   0.78  |
|     cat     |   0.85  |
|    chair    |   0.54  |
|     cow     |   0.66  |
| diningtable |   0.61  |
|     dog     |   0.8   |
|    horse    |   0.78  |
|  motorbike  |   0.82  |
|    person   |   0.81  |
| pottedplant |   0.46  |
|    sheep    |   0.71  |
|     sofa    |   0.69  |
|    train    |   0.84  |
|  tvmonitor  |   0.68  |
+-------------+---------+
```