import torch
from torch import nn

from model.vgg import SeparableConv2d, VGG
from model.loss import MultiBoxLoss


class BoxPredictor(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.num_classes = num_classes
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        for level, (boxes_per_location, out_channels) in enumerate(
                zip([4, 6, 6, 6, 4, 4], (512, 1024, 512, 256, 256, 256))):
            self.cls_headers.append(
                self.cls_block(level, out_channels, boxes_per_location))
            self.reg_headers.append(
                self.reg_block(level, out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        # confidence 映射
        return nn.Conv2d(out_channels,
                         boxes_per_location * self.num_classes,
                         kernel_size=3,
                         stride=1,
                         padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        # box 框回归映射
        return nn.Conv2d(out_channels,
                         boxes_per_location * 4,
                         kernel_size=3,
                         stride=1,
                         padding=1)

    def reset_parameters(self):
        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        cls_logits = []
        bbox_pred = []
        # 根据映射输出 confidence 和 box
        for feature, cls_header, reg_header in zip(features, self.cls_headers,
                                                   self.reg_headers):
            cls_logits.append(
                cls_header(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(
                reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        #转化为向量
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits],
                               dim=1).view(batch_size, -1, self.num_classes)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred],
                              dim=1).view(batch_size, -1, 4)
        return cls_logits, bbox_pred


class SSDBoxHead(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.cfg = num_classes
        self.predictor = BoxPredictor(num_classes=num_classes)
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=3)
        self.priors = None

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        return self.loss(cls_logits, bbox_pred, targets)

    def loss(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred,
                                                 gt_labels, gt_boxes)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict


class SSDDetector(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.backbone = VGG()
        self.box_head = SSDBoxHead(num_classes)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections