import tifffile as tiff
import numpy as np
from pathlib import Path

def convert_to_binary(input_path, output_path=None):
    input_path = Path(input_path)
    masks = tiff.imread(input_path)

    print(f"Loaded mask stack of shape {masks.shape} from {input_path.name}")
    
    binary_masks = (masks > 0).astype(np.uint8)

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_binary.tif")

    tiff.imwrite(output_path, binary_masks, imagej=True)
    print(f"Saved binary mask to {output_path}")
