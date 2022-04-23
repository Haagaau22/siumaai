import pytorch_lightning as pl
from functools import lru_cache
from transformers import AdamW, get_linear_schedule_with_warmup


class Ner(pl.LightningModule):
    def __init__(self, 
            learning_rate, 
            adam_epsilon, 
            warmup_rate, 
            weight_decay,
            model_cls,
            **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls(**model_kwargs)


    # @lru_cache()
    # def total_steps(self):
    #     import ipdb; ipdb.set_trace()
    #     return len(self.train_dataloader()) // self.trainer.accumulate_grad_batches * self.trainer.max_epochs

    def configure_optimizers(self):

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p 
                    for n, p in self.model.named_parameters() 
                    if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p 
                    for n, p in self.model.named_parameters() 
                    if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
                optimizer_grouped_parameters, 
                lr=self.hparams.learning_rate, 
                eps=self.hparams.adam_epsilon)
        # num_train_steps = self.total_steps()
        # num_warmup_steps = int(self.hparams.warmup_rate * num_train_steps)
        scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=int(self.hparams.warmup_rate * self.trainer.estimated_stepping_batches), 
                num_training_steps=self.trainer.estimated_stepping_batches)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def training_step(self, batch, batch_idx):

        loss, *_ = self.model(**batch)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):

        loss, *_ = self.model(**batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)



class CrfNer(Ner):
    def __init__(self, 
            crf_learning_rate,
            learning_rate, 
            adam_epsilon, 
            warmup_rate, 
            weight_decay,
            model_cls,
            **model_kwargs):
        super().__init__(learning_rate, adam_epsilon, warmup_rate, weight_decay, model_cls, **model_kwargs)
        self.save_hyperparameters()

    def configure_optimizers(self):

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p 
                    for n, p in self.model.named_parameters() 
                    if 'crf' in n],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.crf_learning_rate
            },
            {
                "params": [
                    p 
                    for n, p in self.model.named_parameters() 
                    if not any(nd in n for nd in no_decay) and 'crf' not in n],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p 
                    for n, p in self.model.named_parameters() 
                    if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]

        optimizer = AdamW(
                optimizer_grouped_parameters, 
                lr=self.hparams.learning_rate, 
                eps=self.hparams.adam_epsilon)
        # num_train_steps = self.total_steps()
        # num_warmup_steps = int(self.hparams.warmup_rate * num_train_steps)
        scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=int(self.hparams.warmup_rate * self.trainer.estimated_stepping_batches), 
                num_training_steps=self.trainer.estimated_stepping_batches)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
