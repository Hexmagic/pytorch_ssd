import torch
from box_utils import corner_form_to_center_form, convert_boxes_to_locations, assign_priors,center_form_to_corner_form


class SSDTargetTransform:
    def __init__(self, center_form_priors, center_variance, size_variance,
                 iou_threshold):
        # 两种格式先验框
        self.center_form_priors = center_form_priors
        self.corner_form_priors =center_form_to_corner_form(
            center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        # 转换类型
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        #pass
        boxes, labels = assign_priors(gt_boxes, gt_labels,
                                      self.corner_form_priors,
                                      self.iou_threshold)
        # 转换boxes为中心宽高的结构
        boxes = corner_form_to_center_form(boxes)

        locations = convert_boxes_to_locations(boxes, self.center_form_priors,
                                               self.center_variance,
                                               self.size_variance)

        return locations, labels
