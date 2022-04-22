import pytorch_lightning as pl
from layers.fgm import FGM, PGD


class FGM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.fgm = FGM(self.model)
        self.pgd = PGD(self.model)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        assert hasattr(self, 'model')

        loss, *_ = self.model(**batch)
        optimizer.zero_grad()
        self.manual_backward(loss)

        self.fgm.attack()
        loss, *_ = self.model(**batch)
        self.manual_backward(loss)
        self.fgm.restore()

        self.trainer.train_loop.running_loss.append(loss)
        self.log('train_loss', loss, on_epoch=True)
        
        optimizer.step()
        scheduler.step()
        return loss

    def training_step(self, batch, batch_idx):
        K = 3
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        loss, *_ = self.model(**batch)
        optimizer.zero_grad()
        self.manual_backward(loss)
        self.pgd.backup_grad()

        for t in range(K):
            self.pgd.attack(is_first_attack=(t==0))
            if t != K-1:
                optimizer.zero_grad()
            else:
                self.pgd.restore_grad()

            loss, *_ = self.model(**batch)
            self.manual_backward(loss)
        self.pgd.restore()

        self.trainer.train_loop.running_loss.append(loss)
        self.log('train_loss', loss, on_epoch=True)
        
        optimizer.step()
        scheduler.step()
        return loss
