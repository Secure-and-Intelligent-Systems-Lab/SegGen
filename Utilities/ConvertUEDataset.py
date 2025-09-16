import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil

# Define the ground truth colors and their corresponding integer values
colors = [
    ([0, 0, 0], 0),         # Ground
    ([0, 255, 0], 1),       # European Beech
    ([196, 163, 134], 2),   # Rocks
    ([195, 215, 207], 3),   # Norway Maple
    ([97, 44, 32], 4),      # Dead Stuff
    ([29, 124, 0], 5)       # Black Alder
]

def find_closest_class(pixel, colors):
    differences = [np.linalg.norm(np.array(pixel) - np.array(color)) for color, _ in colors]
    return colors[np.argmin(differences)][1]

def smooth_image(im, sigma=3):
    return cv2.medianBlur(im, sigma)

def smooth_labels(label_map):
    kernel = np.ones((3, 3), np.uint8)
    smoothed = cv2.morphologyEx(label_map, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5, 5), np.uint8)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
    return smoothed

def process_image(image_filename):
    global ground_truth_folder, output_folder

    image_path = os.path.join(ground_truth_folder, image_filename)
    if not os.path.isfile(image_path):
        return

    # Read the image using OpenCV
    image = cv2.imread(image_path)[..., :3]
    if image is None:
        print(f"Warning: Failed to read {image_filename}")
        return

    # Convert to RGB
    image = image[:, :, ::-1]

    # Prepare an empty array for the label map
    label_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Assign class labels
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            label_map[i, j] = find_closest_class(pixel, colors)

    # Smooth the labels
    label_map = smooth_labels(label_map)

    # Save the label map to the output folder
    output_path = os.path.join(output_folder, image_filename)
    cv2.imwrite(output_path, label_map)


def rename_and_align_folders(dataset_path):
    folders = ['Depth', 'Optical', 'Labels']
    file_maps = {}

    # Step 1: Collect file names sorted by their numeric values
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        files = sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0]))
        file_maps[folder] = files

    # Step 2: Ensure folders have the same number of files
    min_files = min(len(file_maps['Depth']), len(file_maps['Optical']), len(file_maps['Labels']))

    # Step 3: Rename files to align them from 1 to min_files
    for i in range(min_files):
        for folder in folders:
            old_name = file_maps[folder][i]
            old_path = os.path.join(dataset_path, folder, old_name)
            new_name = f"{i + 1}{os.path.splitext(old_name)[1]}"
            new_path = os.path.join(dataset_path, folder, new_name)

            # Rename file
            shutil.move(old_path, new_path)

    print("Files have been successfully aligned and renamed.")




def main():
    global ground_truth_folder, output_folder
    rename_and_align_folders("/mnt/c/Users/Justin/Downloads/UE5_Dataset_V1/")

    # Paths
    ground_truth_folder = r'/mnt/c/Users/Justin/Downloads/UE5_Dataset_V1/Labels/'
    output_folder = r'/mnt/c/Users/Justin/Downloads/UE5_Dataset_V1/Labels_Processed/'
    os.makedirs(output_folder, exist_ok=True)

    # Get all image filenames
    image_filenames = [f for f in os.listdir(ground_truth_folder) if os.path.isfile(os.path.join(ground_truth_folder, f))]

    # Process images using multiprocessing
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_image, image_filenames), total=len(image_filenames)))

    print(f"Conversion completed. All labeled images are stored in {output_folder}")

if __name__ == "__main__":
    main()
