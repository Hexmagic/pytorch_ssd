from dataset.voc import VOCDataset
from torch.utils.data import DataLoader
from utils.lr_scheduler import make_optimizer, make_lr_scheduler
from model.ssd import SSDDetector
from torch.autograd import Variable


def train():
    model = SSDDetector().cuda()
    optim = make_optimizer(model)
    lr_scheduler = make_lr_scheduler(optim)
    dataloader = DataLoader(VOCDataset(data_dir='datasets', split='train'))
    for img, target in dataloader:
        img, target = Variable(img).cuda(), Variable(target).cuda()
        loss = model(img, target)


if __name__ == '__main__':
    train()