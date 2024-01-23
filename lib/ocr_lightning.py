from typing import Any
import pytorch_lightning as L

import math
import torch
import torch.nn as nn
import torch.optim as optim

from .resnet import ResNet

L.seed_everything(42)
model_dict = {'resnet': ResNet}
def create_model(model_name, model_hparams):
  if model_name in model_dict:
    return model_dict[model_name](**model_hparams)
  else:
    assert False, f"Model {model_name} not found. Available models are: {str(model_dict.keys())}"

def inv_sqrt_decay(lr, warmup_steps, init_lr):
    def _fn(step):
        if step < warmup_steps:
            return init_lr + (lr - init_lr) * (step / warmup_steps)
        else:
            return lr * math.sqrt(warmup_steps / step)

    return _fn

class OCRLit(L.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams, scheduler_hparams=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model_name, model_hparams)
        self.loss_module = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros((1,1,64,64), dtype=torch.float32)
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        # acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        # self.log("train_acc", acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)
    
    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[20000,40000],
            gamma=0.1)
        # we use inv_sqrt_decay scheduler
        # scheduler = optim.lr_scheduler.LambdaLR(
        #     optimizer=optimizer, 
        #     lr_lambda=inv_sqrt_decay(
        #         **self.hparams.scheduler_hparams
        #     ))
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
            "name": "MultiStepLR"
        }
        return [optimizer], [lr_scheduler_config]