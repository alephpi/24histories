import os
import pytorch_lightning as L
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import torch



from lib.ocr_lightning import OCRLit
from lib.dataset import CharDataset

DATASET_PATH = './data/train/'
CHECKPOINT_PATH = './ckpt/recognition/'
NUMBER_OF_CLASSES = 12328

L.seed_everything(42)  # To be reproducible
torch.set_float32_matmul_precision('medium')

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
                filename = "{epoch}-{step}-{val_acc:.8f}",
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

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = OCRLit.load_from_checkpoint(pretrained_filename)
    else:
        model = OCRLit(model_name=model_name, **kwargs)
        datamodule = CharDataset(data_dir='./data', train_batch_size=train_batch_size)
        # tuner = Tuner(trainer)
        # tuner.scale_batch_size(model, mode="power")
        trainer.fit(model=model, datamodule=datamodule)
        model = OCRLit.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    # val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    
    test_result = trainer.test(model=model, datamodule=datamodule, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"]}

    return result
if __name__ == "__main__":
    result = train_model(
        model_name='resnet', 
        # save_name='resnet/lightning_logs/version_15/checkpoints/epoch=1-step=14749-val_acc=0.88',
        train_batch_size = 2240,
        model_hparams={
            'num_classes': NUMBER_OF_CLASSES,
            "c_in": 1,
            "c_hidden": [16,32,64],
            "num_blocks": [3,3,3],
            "act_fn_name": "relu",
            "block_name": "PreActResNetBlock",
        },
        optimizer_name="SGD",
        optimizer_hparams={
            "lr": 0.1,
            # enable for SGD
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