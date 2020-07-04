import torch
from utils.iter_sampler import IterationBasedBatchSampler
from torch.utils.data import DataLoader
from data.voc import VOCDataset


def make_dataloader(dataset, opt):
    batch_size, start_iter, n_cpu, max_iter = 1, 1, 1, 10
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


if __name__ == '__main__':
    dataset = VOCDataset(data_dir='datasets', split='train')
    dataloder = make_dataloader(dataset, None)
    for ele in dataloder:
        print(ele)
