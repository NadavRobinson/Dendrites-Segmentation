import os
import random
import shutil
from pathlib import Path
import cv2

def main():
    # Source paths
    images_src = Path("dataset/Hard")
    labels_src = Path("output/hard_yolo_annotations")
    
    # Destination paths
    yolo_root = Path("dataset/yolo_dataset")
    splits = ["train", "valid", "test"]
    
    # Create labels directories if they don't exist
    for split in splits:
        (yolo_root / split / "labels").mkdir(parents=True, exist_ok=True)
        (yolo_root / split / "images").mkdir(parents=True, exist_ok=True)

    # Get all label files
    label_files = list(labels_src.glob("*.txt"))
    random.seed(42)  # For reproducibility
    random.shuffle(label_files)
    
    # Distribute files
    # With 7 files, let's do 5 train, 1 valid, 1 test
    distribution = ["train"] * 5 + ["valid"] * 1 + ["test"] * 1
    # If there are more than 7, just cycle through
    
    for i, label_file in enumerate(label_files):
        stem = label_file.stem
        # Image can be .tif or .png (from reorganization)
        # Actually in dataset/Hard they are .tif
        image_file = images_src / f"{stem}.tif"
        
        if not image_file.exists():
            print(f"Warning: Image {image_file} not found for label {label_file}")
            continue
            
        split = distribution[i % len(distribution)]
        
        dst_image_dir = yolo_root / split / "images"
        dst_label_dir = yolo_root / split / "labels"
        
        # Copy label
        shutil.copy2(label_file, dst_label_dir / label_file.name)
        
        # Convert image to 3-channel PNG to match existing dataset
        image = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Error: Could not read image {image_file}")
            continue
            
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
        dst_image_path = dst_image_dir / f"{stem}.png"
        cv2.imwrite(str(dst_image_path), image)
        
        print(f"Copied {stem} to {split}")

if __name__ == "__main__":
    main()
