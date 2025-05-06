# extract_centroids_from_tiff.py
# Extract cell centroids from the first frame of a TIFF using Cellpose

from cellpose import models
from tifffile import imread
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from skimage.io import imsave

def segment_centroids(tif_path: Path, diameter=28, model_type='cyto'):
    print(f"[info] loading: {tif_path}")
    stack = imread(str(tif_path))

    # Handle 4D (T, Y, X, RGB) or 3D (Z, Y, X)
    if stack.ndim == 4:
        img = stack[0]  # first timepoint
        if img.shape[-1] == 3:
            img = np.mean(img, axis=2)  # convert RGB to grayscale
    elif stack.ndim == 3:
        if stack.shape[-1] == 3:
            img = np.mean(stack[0], axis=2)
        else:
            img = stack[0]  # single grayscale frame
    else:
        img = stack

    model = models.Cellpose(gpu=False, model_type=model_type)
    masks, _, _, _ = model.eval(img, diameter=diameter, channels=[0, 0], do_3D=False)

    # Save mask overlay for inspection
    plt.imshow(img, cmap='gray')
    plt.imshow(masks, cmap='nipy_spectral', alpha=0.4)
    plt.title(f"Segmentation: {tif_path.name}")
    plt.axis('off')
    overlay_path = tif_path.with_name(tif_path.stem + "_overlay.png")
    plt.savefig(overlay_path, dpi=150)
    plt.close()
    print(f"[info] saved overlay: {overlay_path}")

    # Optional: save raw mask
    mask_path = tif_path.with_name(tif_path.stem + "_masks.tif")
    imsave(mask_path, masks.astype(np.uint16))
    print(f"[info] saved raw masks: {mask_path}")

    centroids = []
    for label in np.unique(masks):
        if label == 0:
            continue
        if masks.ndim == 3:
            _, ys, xs = np.where(masks == label)
        else:
            ys, xs = np.where(masks == label)
        x_mean = np.mean(xs)
        y_mean = np.mean(ys)
        centroids.append((x_mean, y_mean))

    df = pd.DataFrame(centroids, columns=['X', 'Y'])
    df['Movie'] = tif_path.stem
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tiffs", nargs='+', type=Path, help="List of .tif files")
    parser.add_argument("--diameter", type=float, default=28)
    parser.add_argument("--model", type=str, default='cyto')
    parser.add_argument("--out", type=Path, default=Path("centroids.csv"))
    args = parser.parse_args()

    dfs = [segment_centroids(tif, args.diameter, args.model) for tif in args.tiffs]
    out_df = pd.concat(dfs, ignore_index=True)
    out_df.to_csv(args.out, index=False)
    print(f"[done] saved centroids to {args.out}")
