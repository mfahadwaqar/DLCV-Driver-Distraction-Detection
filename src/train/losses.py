"""Missing-label masking shared by baseline and later multi-task training."""

import torch.nn.functional as F


def masked_cross_entropy(logits, labels, weight=None):
    valid = labels != -1
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid], weight=weight)
