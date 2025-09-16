import os
import cv2
import numpy as np

# Define class index to color mapping
colors = [
    ([0, 0, 0], 0),         # Ground
    ([0, 255, 0], 1),       # European Beech
    ([196, 163, 134], 2),   # Rocks
    ([195, 215, 207], 3),   # Norway Maple
    ([97, 44, 32], 4),      # Dead Stuff
    ([29, 124, 0], 5)       # Black Alder
]

# Create a lookup array for fast mapping
idx_to_color = np.zeros((256, 3), dtype=np.uint8)
for color, idx in colors:
    idx_to_color[idx] = color

def convert_to_rgb(mask):
    """Convert grayscale class index mask to RGB image using the palette."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in np.unique(mask):
        if idx < len(idx_to_color):
            rgb[mask == idx] = idx_to_color[idx]
    return rgb

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load {input_path}")
                continue

            rgb_img = convert_to_rgb(mask)
            cv2.imwrite(output_path, rgb_img)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_dir = "/home/justin/PycharmProjects/Datasets/UE5_Dataset_V1/Labels_Processed"   # Change to your grayscale mask folder
    output_dir = "/home/justin/PycharmProjects/Datasets/UE5_Dataset_V1/output_rgb"    # Desired RGB output folder
    process_folder(input_dir, output_dir)
