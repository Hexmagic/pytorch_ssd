import collections
import numpy as np
import itertools
from typing import List
from math import sqrt
SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])

Spec = collections.namedtuple(
    'Spec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])

# the SSD orignal specs
specs = [
    Spec(38, 8, SSDBoxSizes(30, 60), [2]),
    Spec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
    Spec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
    Spec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
    Spec(3, 100, SSDBoxSizes(213, 264), [2]),
    Spec(1, 300, SSDBoxSizes(264, 315), [2])
]



def generate_ssd_prios(specs: List[Spec], image_size=300, clip=True):
    boxes = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale
            size = spec.box_sizes.min
            h = w = size / image_size
            boxes.append([x_center, y_center, h, w])
            size = np.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            h = w = size / image_size
            boxes.append([x_center, y_center, h, w])
            size = spec.box_sizes.min
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = sqrt(ratio)
                boxes.append([x_center, y_center, h * ratio, w / ratio])
                boxes.append([x_center, y_center, h / ratio, w * ratio])
    boxes = np.array(boxes)
    if clip:
        boxes = np.clip(boxes, 0.0, 1.0)
    return boxes

generate_ssd_prios(specs)