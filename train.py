import os
from h11 import Data
import pytorch_lightning as L
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split


import torch
import numpy as np
import cv2
from collections import Counter
from sklearn.model_selection import train_test_split



from lib.ocr_lightning import OCRLit

DATASET_PATH = './data/train/'
CHECKPOINT_PATH = './ckpt/recognition/'
NUMBER_OF_CLASSES = 12328

torch.set_float32_matmul_precision('medium')

def stratified_split(dataset, ratio):
    X = np.array([i for i in range(len(dataset))])
    y = np.array(dataset.targets)
    train_idx, val_idx, _, _ = train_test_split(X,y, test_size=ratio, stratify=y)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    return train_dataset, val_dataset

def load_dataset(train_batch_size):
    train_val_set = datasets.ImageFolder('./data/train/', 
                               loader=lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), 
                               transform=transforms.ToTensor())
    test_set = datasets.ImageFolder('./data/test/', 
                               loader=lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), 
                               transform=transforms.ToTensor())

    # train_dataset, val_dataset = random_split(charset, [0.9, 0.1])
    train_set, val_set = stratified_split(train_val_set, 0.1)
    # a = dict(Counter(charset.targets[i] for i in train_dataset.indices))
    # print(max(a.values()),min(a.values()))

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=12)
    return train_loader, val_loader, test_loader

def train_model(model_name, train_batch_size, save_name=None, **kwargs):
    """Train model.

    Args:
        model_name: Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional): If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models
        # We run on a single GPU (if possible)
        accelerator="auto",
        devices=1,
        precision='16-mixed',
        # How many epochs to train for if no patience is set
        max_epochs=10,
        val_check_interval=1000,
        callbacks=[
            # ModelCheckpoint(
            #     save_weights_only=True, mode="max", monitor="val_acc"
            # ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            ModelCheckpoint(
                filename = "{epoch}-{step}-{val_acc:.2f}",
                save_top_k=10,
                monitor="val_acc",
                mode='max',
            ),
            LearningRateMonitor("step"),
            EarlyStopping(
                monitor='val_acc',
                min_delta=0,
                patience=5,
                mode="max",
                strict=True
            )
        ],  # Log learning rate every epoch
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    trainer.logger.log_hyperparams({"train_batch_size": train_batch_size})

    # load dataset
    train_loader, val_loader, test_loader = load_dataset(train_batch_size=train_batch_size)
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = OCRLit.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducible
        model = OCRLit(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = OCRLit.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    # val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"val": test_result[0]["test_acc"]}

    return result
if __name__ == "__main__":
    result = train_model(
        model_name='resnet', 
        save_name='resnet/lightning_logs/version_15/checkpoints/epoch=1-step=14749-val_acc=0.88',
        train_batch_size = 256,
        model_hparams={
            'num_classes': NUMBER_OF_CLASSES,
            "c_in": 1,
            "c_hidden": [8,16,32],
            "num_blocks": [2,2,2],
            "act_fn_name": "relu",
            "block_name": "PreActResNetBlock",
        },
        optimizer_name="SGD",
        optimizer_hparams={
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4,
        },
        # scheduler_hparams={
        #     "lr": 0.1,
        #     "init_lr": 1e-4,
        #     "warmup_steps": 1000,

        # }
        )
    print(result)