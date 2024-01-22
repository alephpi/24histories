import os
import pytorch_lightning as L
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import cv2

from lib.ocr_lightning import OCRLit

DATASET_PATH = './data/train/'
CHECKPOINT_PATH = './ckpt/recognition/'
NUMBER_OF_CLASSES = 12328

torch.set_float32_matmul_precision('medium')

def load_dataset():
    charset = datasets.ImageFolder('./data/train/', 
                               loader=lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), 
                               transform=transforms.ToTensor())
    train_dataset, val_dataset = random_split(charset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=12)
    return train_loader, val_loader

def train_model(model_name, save_name=None, **kwargs):
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
        # How many epochs to train for if no patience is set
        max_epochs=20000,
        callbacks=[
            # ModelCheckpoint(
            #     save_weights_only=True, mode="max", monitor="val_acc"
            # ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            ModelCheckpoint(
                filename = "checkpoint_{epoch:03d}",
                every_n_epochs=500,
            ),
            LearningRateMonitor("epoch"),
        ],  # Log learning rate every epoch
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = OCRLit.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducible
        train_loader, val_loader = load_dataset()
        model = OCRLit(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = OCRLit.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    # test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"val": val_result[0]["test_acc"]}

    return result
if __name__ == "__main__":
    result = train_model(
        model_name='resnet', 
        model_hparams={
            'num_classes': NUMBER_OF_CLASSES,
            "c_in": 1,
            "c_hidden": [8,16,32,64],
            "num_blocks": [3,4,6,3],
            "act_fn_name": "relu",
            "block_name": "PreActResNetBlock",
        },
        optimizer_name="SGD",
        optimizer_hparams={
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4,
        })
    print(result)