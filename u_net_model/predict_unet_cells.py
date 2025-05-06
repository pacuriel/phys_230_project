import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A

class CellSegmentationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.filenames = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, axis=-1)

        if self.transform:
            image = self.transform(image=image)['image']

        image = image.astype("float32") / 255.0
        return torch.tensor(image).permute(2, 0, 1), self.filenames[idx]

def predict():
    image_dir = "synthetic_images"
    out_dir = Path("predicted_masks")
    out_dir.mkdir(exist_ok=True)

    transform = A.Compose([A.Resize(256, 256)])
    dataset = CellSegmentationDataset(image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load("unet_cell_segmentation.pth", map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        for img, fname in tqdm(loader, desc="Predicting"):
            pred = model(img)
            mask = torch.sigmoid(pred).squeeze().numpy()
            mask = (mask > 0.5).astype(np.uint8) * 255
            cv2.imwrite(str(out_dir / fname[0].replace("frame_", "pred_")), mask)

    print("✅ Saved predicted masks to", out_dir.resolve())

if __name__ == "__main__":
    predict()