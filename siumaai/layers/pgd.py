import torch
from torch import nn

class PGD:
    def __init__(self, model: nn.Module, eps=1.0, alpha=0.3):
        if hasattr(model, 'module'):
            self.model = model.module
        else:
            self.model = model
        self.eps = eps
        self.alpha = alpha
        self.embedding_backup = {}
        self.grad_backup = {}

    def attack(self, embedding_name='word_embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embedding_name in name:
                if is_first_attack is True:
                    self.embedding_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)

                if norm and not torch.isnan(norm):
                    param.data.add_(self.alpha * param.grad / norm)
                    param.data = self.project(name, param.data)

    def restore(self, embedding_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embedding_name in name:
                assert name in self.embedding_backup, [name, self.embedding_backup.keys()]
                param.data = self.embedding_backup[name]
        self.embedding_backup = {}

    def project(self, name, param_data):
        r = param_data - self.embedding_backup[name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.embedding_backup[name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
                 
