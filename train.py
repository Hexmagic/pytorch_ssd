from dataset.voc import VOCDataset
from torch.utils.data import DataLoader
from utils.lr_scheduler import make_optimizer, make_lr_scheduler
from model.ssd import SSDDetector
from torch.autograd import Variable
import numpy as np
from argparse import ArgumentParser
import os
import torch
import time


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def train():
    model = SSDDetector().cuda()
    optim = make_optimizer(model)
    lr_scheduler = make_lr_scheduler(optim)

    total_loss, reg_losses, cls_losses = [], [], []
    i = 0
    parser = ArgumentParser()
    parser.add_argument('--iters', type=int, default=120000)
    parser.add_argument('--start_iter', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='weights')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='datasets')
    opt = parser.parse_args()
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    data_set = VOCDataset(data_dir=opt.data_dir, split='train')
    dataloader = DataLoader(data_set,
                            batch_size=opt.batch_size,
                            pin_memory=True,
                            collate_fn=data_set.collate_fn,
                            num_workers=8)
    data_iter = iter(dataloader)
    start = time.time()
    for iter_i in range(opt.start_iter, opt.iters):
        try:
            img, target, _ = next(data_iter)
        except:
            data_iter = iter(dataloader)
            img, target, _ = next(data_iter)
        i += 1
        memory = torch.cuda.max_memory_allocated() // 1024 // 1024
        img = Variable(img).cuda()
        for key in target.keys():
            target[key] = Variable(target[key]).cuda()
        loss_dict = model(img, target)
        loss = sum(loss for loss in loss_dict.values())
        reg_losses.append(loss_dict['reg_loss'].item())
        cls_losses.append(loss_dict['cls_loss'].item())
        total_loss.append(loss.item())
        #loss_dict = reduce_loss_dict(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_scheduler.step()
        if i % 2000 == 0:
            torch.save(model, f"{opt.save_path}/{iter_i}_ssd300.pth")
        if i % 10 == 0:
            end = time.time()
            eta = round(end - start, 2)
            print(
                f"iter {iter_i} loss total {np.mean(total_loss).round(2)} reg {np.mean(reg_losses).round(2)} cls {np.mean(cls_losses).round(2)} Mem {memory} M ETA: {eta}"
            )
            start = end

    torch.save(model, f"{opt.save_path}/ssd300_final.pth")


if __name__ == '__main__':
    train()