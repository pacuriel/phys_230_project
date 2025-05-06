import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def compute_metrics(mask_dir, pred_dir):
    mask_files = sorted(Path(mask_dir).glob("mask_*.png"))
    pred_files = sorted(Path(pred_dir).glob("pred_*.png"))

    dice_scores = []
    iou_scores = []
    pixel_accuracies = []

    for mask_file, pred_file in tqdm(zip(mask_files, pred_files), total=len(mask_files), desc="Evaluating"):
        gt = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 0
        pred_raw = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
        pred = cv2.resize(pred_raw, (gt.shape[1], gt.shape[0])) > 0

        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        gt_sum = gt.sum()
        pred_sum = pred.sum()

        dice = (2.0 * intersection) / (gt_sum + pred_sum + 1e-8)
        iou = intersection / (union + 1e-8)
        accuracy = (gt == pred).sum() / gt.size

        dice_scores.append(dice)
        iou_scores.append(iou)
        pixel_accuracies.append(accuracy)

    print("✅ Evaluation Summary:")
    print(f"Mean Dice Coefficient: {np.mean(dice_scores):.4f}")
    print(f"Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"Mean Pixel Accuracy: {np.mean(pixel_accuracies):.4f}")

if __name__ == "__main__":
    compute_metrics("synthetic_masks", "predicted_masks")