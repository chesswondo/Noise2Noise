import torch
import matplotlib.pyplot as plt
import numpy as np
import hydra
from hydra.utils import instantiate
import argparse

from src.data_module import DenoisingDataModule
from src.lightning_module import DenoisingModule

def visualize(checkpoint_path, config_path="configs", num_samples=4, val_dir="data/valid"):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path=config_path)
    cfg = hydra.compose(config_name="config")
    
    unet = instantiate(cfg.model)
    model = DenoisingModule(model=unet, lr=cfg.training.lr, perceptual_weight=cfg.training.perceptual_weight)

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(
        checkpoint_path,
        # map_location=torch.device('cpu')
    )
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    
    model.eval()
    model.freeze()

    dm = DenoisingDataModule(train_dir="", val_dir=val_dir, patch_size=256)
    dm.setup(stage=None)
    val_loader = dm.val_dataloader()

    batch = next(iter(val_loader))
    noisy_inputs, _, clean_targets = batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    noisy_inputs = noisy_inputs.to(device)

    with torch.no_grad():
        denoised_outputs = model(noisy_inputs)

    noisy_inputs = noisy_inputs.cpu()
    denoised_outputs = denoised_outputs.cpu()

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    plt.tight_layout()

    for i in range(num_samples):
        # [C, H, W] -> [H, W, C]
        noisy_img = noisy_inputs[i].permute(1, 2, 0).cpu().numpy()
        denoised_img = denoised_outputs[i].permute(1, 2, 0).cpu().numpy()
        clean_img = clean_targets[i].permute(1, 2, 0).cpu().numpy()

        denoised_img = np.clip(denoised_img, 0, 1)

        ax = axes[i] if num_samples > 1 else axes
        
        ax[0].imshow(noisy_img)
        ax[0].set_title("Noisy Input")
        ax[0].axis("off")

        ax[1].imshow(denoised_img)
        ax[1].set_title("Model Prediction")
        ax[1].axis("off")

        ax[2].imshow(clean_img)
        ax[2].set_title("Clean Ground Truth")
        ax[2].axis("off")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file"
    )

    args = parser.parse_args()
    visualize(args.checkpoint)