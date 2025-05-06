import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import pandas as pd

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CellSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.filenames[idx].replace("frame_", "mask_"))
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Normalize each image independently
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1) // 255

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.permute(2, 0, 1).float()

def dice_bce_loss(pred, target):
    dice = smp.losses.DiceLoss(mode="binary")(pred, target)
    bce = smp.losses.SoftBCEWithLogitsLoss()(pred, target)
    return dice + bce

def main():
    image_dir = "synthetic_images"
    mask_dir = "synthetic_masks"
    transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = CellSegmentationDataset(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    losses = []
    for epoch in range(20):
        total_loss = 0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            pred = model(x)
            loss = dice_bce_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append({"epoch": epoch+1, "train_loss": avg_loss})
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), "unet_cell_segmentation.pth")
    print("✅ Model saved to unet_cell_segmentation.pth")
    pd.DataFrame(losses).to_csv("loss_log.csv", index=False)
    print("📈 Training log saved to loss_log.csv")

if __name__ == "__main__":
    main()