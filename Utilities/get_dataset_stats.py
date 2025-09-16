import os
import numpy as np
import cv2
from collections import defaultdict
from typing import Dict, Tuple

def compute_class_distribution(label_dir: str, train_txt: str, num_classes: int = 7) -> Tuple[Dict[int, float], Dict[int, Tuple[float, float]]]:
    """
    Args:
        label_dir (str): Path to the folder with label images
        train_txt (str): Path to the train.txt file
        num_classes (int): Number of semantic classes (0-indexed)
        image_ext (str): File extension of label images

    Returns:
        total_class_percent (dict): % of each class across the entire dataset
        per_image_stats (dict): mean and std of % presence per class across images
    """
    with open(train_txt, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]

    # Per-class pixel counts (global and per-image)
    total_pixels_per_class = np.zeros(num_classes, dtype=np.int64)
    per_image_class_percentages = []

    for name in image_names:
        label_path = os.path.join(label_dir, name)
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if label_img is None:
            print(f"Warning: {label_path} could not be loaded.")
            continue

        total_pixels = label_img.size
        class_counts = np.bincount(label_img.flatten(), minlength=num_classes)
        total_pixels_per_class += class_counts

        class_percents = (class_counts / total_pixels) * 100.0
        per_image_class_percentages.append(class_percents)

    # Global distribution
    total_pixels_all = total_pixels_per_class.sum()
    total_class_percent = {
        cls: round((count / total_pixels_all) * 100, 4)
        for cls, count in enumerate(total_pixels_per_class)
    }

    # Per-image stats
    per_image_class_percentages = np.stack(per_image_class_percentages)  # shape: (num_images, num_classes)
    per_image_stats = {
        cls: (
            round(np.mean(per_image_class_percentages[:, cls]), 4),
            round(np.std(per_image_class_percentages[:, cls]), 4)
        )
        for cls in range(num_classes)
    }

    return total_class_percent, per_image_stats

# Example usage:
if __name__ == "__main__":
    label_folder = "/home/justin/PycharmProjects/Datasets/UE5_Dataset_V1/Labels_Processed"
    train_file = "/home/justin/PycharmProjects/Datasets/UE5_Dataset_V1/train.txt"
    num_classes = 6  # Adjust this based on your dataset

    total_dist, per_img_stats = compute_class_distribution(label_folder, train_file, num_classes)

    print("📊 Total Class Distribution (%):")
    for cls, pct in total_dist.items():
        print(f"Class {cls}: {pct}%")

    print("\n📈 Per-Image Class Distribution Stats (mean %, std %):")
    for cls, (mean_pct, std_pct) in per_img_stats.items():
        print(f"Class {cls}: Mean = {mean_pct}%, Std = {std_pct}%")
