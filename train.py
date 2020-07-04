from data.voc import VOCDataset
from torch.utils.data import DataLoader
from utils.lr_scheduler import make_optimizer, make_lr_scheduler
from model.ssd import SSDDetector
from torch.autograd import Variable
import numpy as np
from argparse import ArgumentParser
import os
import torch
import time
from utils.iter_sampler import IterationBasedBatchSampler
from inference import do_evaluation
from prettytable import PrettyTable


def make_dataloader(dataset, opt):
    batch_size, start_iter, n_cpu, max_iter = opt.batch_size, opt.start_iter, opt.n_cpu, opt.max_iter
    sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler=sampler, batch_size=batch_size, drop_last=False)
    if max_iter is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler,
                                                   num_iterations=max_iter,
                                                   start_iter=start_iter)

    data_loader = DataLoader(dataset,
                             num_workers=n_cpu,
                             batch_sampler=batch_sampler,
                             pin_memory=True)
    return data_loader


def train():
    model = SSDDetector().cuda()
    optim = make_optimizer(model)
    lr_scheduler = make_lr_scheduler(optim)

    total_loss, reg_losses, cls_losses = [], [], []
    i = 0
    parser = ArgumentParser()
    parser.add_argument('--max_iter', type=int, default=120000)
    parser.add_argument('--start_iter', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='weights')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='datasets')
    parser.add_argument('--n_cpu', type=int, default=8, help='num workers')
    opt = parser.parse_args()
    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    dataset = VOCDataset(data_dir=opt.data_dir, split='train')
    dataloader = make_dataloader(dataset, opt)
    start = time.time()
    for iter_i, (img, target, _) in enumerate(dataloader):
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
        if iter_i % 5000 == 0:
            rst = do_evaluation(model)
            table = PrettyTable()
            table.add_row(['类别', '值'])
            for k, v in rst[0].items():
                table.add_column([k, rount(v, 2)])
            print(table)
            torch.save(model, f"{opt.save_path}/{iter_i}_ssd300.pth")
        if iter_i % 10 == 0:
            memory = torch.cuda.max_memory_allocated() // 1024 // 1024
            end = time.time()
            eta = round(end - start, 2)
            print(
                f"iter {iter_i} loss total {np.mean(total_loss).round(2)} reg {np.mean(reg_losses).round(2)} cls {np.mean(cls_losses).round(2)} Mem {memory} M ETA: {eta}"
            )
            start = end

    torch.save(model, f"{opt.save_path}/ssd300_final.pth")


if __name__ == '__main__':
    train()