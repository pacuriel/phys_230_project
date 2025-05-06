import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from final_project_phys230.data_finalproj.u_net_model.train_unet_cells2 import CellSegmentationDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

def visualize_predictions(image_dir="synthetic_images", mask_dir="synthetic_masks",
                          model_path="unet_cell_segmentation.pth", out_dir="viz_preds", num_examples=5):
    os.makedirs(out_dir, exist_ok=True)

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = CellSegmentationDataset(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    for i, (image, mask) in enumerate(tqdm(dataloader, total=num_examples)):
        if i >= num_examples:
            break
        with torch.no_grad():
            pred = model(image).sigmoid().squeeze().cpu().numpy()
        image_np = image.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image_np, cmap="gray")
        axs[0].set_title("Input")
        axs[1].imshow(mask_np, cmap="gray")
        axs[1].set_title("Ground Truth")
        axs[2].imshow(pred > 0.5, cmap="gray")
        axs[2].set_title("Prediction")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        out_path = Path(out_dir) / f"viz_{i}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    visualize_predictions()