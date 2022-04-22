import torch
from torch import nn


class FGM:
    def __init__(self, model: nn.Module, eps=1.0):
        if hasattr(model, 'module'):
            self.model = model.module
        else:
            self.model = model
        self.eps = eps
        self.backup = {}

    def attack(self, embedding_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embedding_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    param.data.add_(self.eps * param.grad / norm)

    def restore(self, embedding_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embedding_name in name:
                assert name in self.backup, [name, self.backup.keys()]
                param.data = self.backup[name]
        self.backup = {}
