from torch import nn
from torch.optim import AdamW, SGD


def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01):
    if optimizer == 'adamw':
        return AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(filter(lambda p: p.requires_grad, model.parameters()), lr, momentum=0.9, weight_decay=weight_decay)