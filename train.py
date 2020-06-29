from dataset.voc import VOCDataset
from torch.utils.data import DataLoader
from utils.lr_scheduler import make_optimizer, make_lr_scheduler
from model.ssd import SSDDetector
from torch.autograd import Variable
import numpy as np
from argparse import ArgumentParser
import os


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

    losses = []
    i = 0
    parser = ArgumentParser()
    parser.add_argument('--iters', type=int, default=120000)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='weights')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='datasets')
    opt = parser.parse_args()
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    dataloader = DataLoader(VOCDataset(data_dir=opt.data_dir, split='train'),
                            batch_size=opt.batch_size)
    data_iter = iter(dataloader)
    for iter_i in range(opt.start_iter, opt.iters):
        try:
            img, target = next(data_iter)
        except:
            data_iter = iter(dataloader)
            img, target = next(data_iter)
        i += 1
        img = Variable(img).cuda()
        for key in target.keys():
            target[key] = Variable(target[key]).cuda()
        loss_dict = model(img, target)
        loss = sum(loss for loss in loss_dict.values())
        losses.append(loss.item())
        #loss_dict = reduce_loss_dict(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_scheduler.step()
        if i % 10 == 0:
            print(f"iter {iter_i} loss {np.mean(losses)}")
        if i % 2000 == 0:
            torch.save(model, f"{opt.save_path}/{iter_i}_ssd300.pth")
    torch.save(model, f"{opt.save_path}/ssd300_final.pth")


if __name__ == '__main__':
    train()