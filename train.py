import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.lightning_module import DenoisingModule

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.training.seed)

    print(f"Instantiating DataModule: {cfg.data._target_}")
    dm = instantiate(cfg.data)

    print(f"Instantiating Model Architecture: {cfg.model._target_}")
    unet_architecture = instantiate(cfg.model)

    model = DenoisingModule(
        model=unet_architecture,
        lr=cfg.training.lr,
        perceptual_weight=cfg.training.perceptual_weight
    )

    wandb_logger = WandbLogger(
        project=cfg.training.project_name,
        name=cfg.training.experiment_name,
        log_model=False
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint-{epoch:02d}-{val_psnr:.2f}",
        monitor="val_psnr",
        mode="max",
        save_top_k=1
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)
    wandb.finish()



if __name__ == "__main__":
    train()