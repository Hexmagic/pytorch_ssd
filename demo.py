import glob
import os
import time

import torch
from PIL import Image
from vizer.draw import draw_boxes
from model.ssd import SSDDetector

from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer


@torch.no_grad()
def run_demo(ckpt, score_threshold, images_dir, output_dir):
    class_names = VOCDataset.class_names
    model = torch.load(ckpt)
    transforms = build_transforms(cfg, is_train=False)
    model.eval()
    for i, image_path in enumerate('sample'):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.cuda())[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).cpu().numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result[
            'scores']

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = ' | '.join([
            'objects {:02d}'.format(len(boxes)),
            'load {:03d}ms'.format(round(load_time * 1000)),
            'inference {:03d}ms'.format(round(inference_time * 1000)),
            'FPS {}'.format(round(1.0 / inference_time))
        ])
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths),
                                              image_name, meters))

        drawn_image = draw_boxes(image, boxes, labels, scores,
                                 class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument("--ckpt",
                        type=str,
                        default='model_047500.pth',
                        help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--images_dir",
                        default='sample',
                        type=str,
                        help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help='Specify a image dir to save predicted images.')
    parser.add_argument(
        "--dataset_type",
        default="voc",
        type=str,
        help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)
    print("Running with config:\n{}".format(cfg))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    run_demo(ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir)


if __name__ == '__main__':
    main()
