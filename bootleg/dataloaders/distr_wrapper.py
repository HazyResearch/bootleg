# https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143

import torch.utils.data

class DistributedIndicesWrapper(torch.utils.data.Dataset):
    """
    Utility wrapper so that torch.utils.data.distributed.DistributedSampler can work with train test splits
    """
    def __init__(self, dataset: torch.utils.data.Dataset, indices: torch.Tensor):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return self.indices.size(0)

    def __getitem__(self, item):
        idx = self.indices[item]
        return self.dataset[idx]