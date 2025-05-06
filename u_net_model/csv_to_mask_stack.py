import pandas as pd
import numpy as np
from skimage.draw import disk
from tifffile import imwrite
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def csv_to_mask_stack(csv_path, output_folder, stack_path=None, img_shape=(600, 600), radius=6):
    # Load tracks
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["X", "Y", "Frame"])
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    frames = sorted(df["Frame"].unique())
    print(f"📊 Loaded {len(frames)} frames from {csv_path}")

    stack_path = Path(stack_path) if stack_path else None
    if stack_path:
        temp_stack = []

    for frame in tqdm(frames, desc="Generating masks"):
        mask = np.zeros(img_shape, dtype=np.uint8)
        frame_df = df[df["Frame"] == frame]

        for _, row in frame_df.iterrows():
            rr, cc = disk((row["Y"], row["X"]), radius, shape=img_shape)
            mask[rr, cc] = 1

        # Save PNG
        png_path = output_folder / f"mask_{int(frame):03d}.png"
        Image.fromarray(mask * 255).save(png_path)

        # Save to stack (optional)
        if stack_path:
            temp_stack.append(mask)

    if stack_path:
        print(f"💾 Writing full multi-page TIFF to {stack_path}")
        imwrite(str(stack_path), np.stack(temp_stack), imagej=True)

    print("✅ Done.")


# Example usage
if __name__ == "__main__":
    csv_to_mask_stack(
        csv_path="all_tracks.csv",
        output_folder="synthetic_masks",
        stack_path="synthetic_masks_stack.tif"
    )
