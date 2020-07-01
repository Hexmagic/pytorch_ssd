import logging
import os

import torch
import torch.utils.data
from tqdm import tqdm
from torch.autograd import Variable

from data.voc import VOCDataset
from utils.dist_util import all_gather, is_main_process, synchronize
from utils.evaluation import voc_evaluation


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("SSD.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def compute_on_dataset(model, data_loader):
    results_dict = {}
    for batch in tqdm(data_loader):
        images, targets, image_ids = batch
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            outputs = model(images.to('cuda'))
            for output in outputs:
                for key in output.keys():
                    output[key] = Variable(output[key]).cuda()
            #outputs = [o.to(cpu_device) for o in outputs]
        results_dict.update(
            {img_id: result
             for img_id, result in zip(image_ids, outputs)})
    return results_dict


def inference(model, data_loader, output_folder='output'):
    dataset = data_loader.dataset
    predictions_path = os.path.join(output_folder, 'predictions.pth')
    predictions = compute_on_dataset(model, data_loader)
    synchronize()
    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if output_folder:
        torch.save(predictions, predictions_path)
    return voc_evaluation(dataset=dataset,
                          predictions=predictions,
                          output_dir=output_folder)


@torch.no_grad()
def do_evaluation(model):
    from torch.utils.data import DataLoader
    model.eval()
    data_loader = DataLoader(VOCDataset('datasets', split='val'))
    eval_results = []
    eval_result = inference(model, data_loader)
    eval_results.append(eval_result)
    return eval_results


if __name__ == '__main__':
    model = torch.load('weights/86000_ssd300.pth')
    rst = do_evaluation(model)
    print(rst)
