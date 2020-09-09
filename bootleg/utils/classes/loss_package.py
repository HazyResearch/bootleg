"""Defines the package of information that is passed back from the scorer"""
from collections import defaultdict

import torch


class LossPackage:
    def __init__(self, device):
        self.loss_dict = {}
        self.loss = torch.zeros(1,1).to(device)

    # add loss to total loss and sets the loss to the associated key
    def add_loss(self, key, loss):
        assert key not in self.loss_dict
        assert type(loss) is torch.Tensor
        self.loss_dict[key] = loss.data.item()
        self.loss += loss

    def merge_loss_packages(self, other_loss):
        num_total_keys = len(other_loss.loss_dict.keys()) + len(self.loss_dict.keys())
        self.loss += other_loss.loss
        self.loss_dict.update(other_loss.loss_dict)
        assert num_total_keys == len(self.loss_dict.keys()), 'Loss dicts were not merged correctly. You may have overlapping loss keys'

    def __repr__(self):
        res = []
        for k in self.loss_dict:
            res.append(f"{k}: {self.loss_dict[k]}")
        return " ".join(res)