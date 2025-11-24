import torch
import matplotlib.pyplot as plt
import numpy as np
import hydra
from hydra.utils import instantiate
import argparse
from PIL import Image
from torchvision import transforms

# We don't need DenoisingDataModule anymore for a single image inference
from src.lightning_module import DenoisingModule

def visualize_single(checkpoint_path, image_path, config_path="configs", noise_factor=0.1):
    """
    Loads a model, loads an image, adds noise, and runs inference.
    Returns: noisy_img, prediction, clean_img (as numpy arrays)
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path=config_path)
    cfg = hydra.compose(config_name="config")
    
    unet = instantiate(cfg.model)
    model = DenoisingModule(model=unet, lr=cfg.training.lr, perceptual_weight=cfg.training.perceptual_weight)

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.freeze()

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts [0, 255] to [0.0, 1.0]
        transforms.Resize((256, 256)),
    ])
    
    clean_tensor = transform(img).to(device)
    
    noise = torch.randn_like(clean_tensor) * noise_factor
    noisy_tensor = clean_tensor + noise
    noisy_tensor = torch.clamp(noisy_tensor, 0., 1.)

    input_batch = noisy_tensor.unsqueeze(0)
    
    with torch.no_grad():
        denoised_batch = model(input_batch)

    noisy_output = noisy_tensor.permute(1, 2, 0).cpu().numpy()
    denoised_output = denoised_batch[0].permute(1, 2, 0).cpu().numpy()
    clean_output = clean_tensor.permute(1, 2, 0).cpu().numpy()

    denoised_output = np.clip(denoised_output, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.tight_layout()

    axes[0].imshow(noisy_output)
    axes[0].set_title("Noisy Input")
    axes[0].axis("off")

    axes[1].imshow(denoised_output)
    axes[1].set_title("Model Prediction")
    axes[1].axis("off")

    axes[2].imshow(clean_output)
    axes[2].set_title("Clean Ground Truth")
    axes[2].axis("off")

    plt.show()

    return noisy_output, denoised_output, clean_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model checkpoint on a single image.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the source image file"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Standard deviation of Gaussian noise to add (default: 0.1)"
    )

    args = parser.parse_args()
    visualize_single(args.checkpoint, args.image, noise_factor=args.noise)