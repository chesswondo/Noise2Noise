import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import wandb

from .losses.perceptual_loss import VGGPerceptualLoss

class DenoisingModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float = 1e-3, perceptual_weight: float = 0.1, image_log_every_epochs: int = 10,
                 max_images: int = 4, image_size: tuple = (128, 128), log_images: bool = True):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.perceptual_loss_fn = VGGPerceptualLoss().eval()
        self.perceptual_weight = perceptual_weight

        self.image_log_every_epochs = image_log_every_epochs
        self.max_images = max_images
        self.image_size = image_size
        self.log_images = log_images

    def _prep_img_for_wandb(self, img_tensor: torch.Tensor) -> np.ndarray:
        img = img_tensor.unsqueeze(0)  # 1,C,H,W
        img = F.interpolate(img, size=self.image_size, mode='bilinear', align_corners=False)
        img = img.squeeze(0)  # C,H,W
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        return img
        
    def forward(self, x):
        return self.model(x)

    def calc_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pixel_loss = F.l1_loss(pred, target)
        perc_loss = self.perceptual_loss_fn(pred, target)
        
        total_loss = pixel_loss + (self.perceptual_weight * perc_loss)
        
        self.log("loss/pixel", pixel_loss)
        self.log("loss/perceptual", perc_loss)
        
        return total_loss

    def _calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return 100.0
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def training_step(self, batch, batch_idx):
        noisy_input, noisy_target, _ = batch 

        prediction = self(noisy_input)
        
        loss = self.calc_loss(prediction, noisy_target)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy_input, noisy_target, clean_gt = batch
        
        prediction = self(noisy_input)
        
        val_loss = self.calc_loss(prediction, noisy_target)
        psnr = self._calculate_psnr(prediction, clean_gt)
        
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr, on_epoch=True, prog_bar=True)

        if not self.log_images:
            return
        
        # Log images only every N epochs and only for the first batch
        if (batch_idx == 0) and (self.current_epoch % self.image_log_every_epochs == 0):
            n = min(noisy_input.size(0), self.max_images)
            columns = ["Original (Clean)", "Noisy Input", "Model Prediction"]
            data = []

            for i in range(n):
                clean = self._prep_img_for_wandb(clean_gt[i])
                noisy = self._prep_img_for_wandb(noisy_input[i])
                pred = self._prep_img_for_wandb(prediction[i])

                data.append([
                    wandb.Image(clean, caption=f"Clean (epoch {self.current_epoch})"),
                    wandb.Image(noisy, caption="Noisy"),
                    wandb.Image(pred, caption="Denoised")
                ])

            try:
                self.logger.log_table(key="Validation_Samples", columns=columns, data=data)
            except Exception:
                run = getattr(self.logger, "experiment", None)
                if run:
                    table = wandb.Table(columns=columns, data=data)
                    run.log({"Validation_Samples": table}, commit=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            }
        }