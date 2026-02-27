import cv2
import numpy as np

cutoff_value = 120

# 1. Load your binary mask
file = 'easy_2'
inout_folder = 'maked_dataset/Easy/'
output_folder = 'annotations/Easy/'
image_path = f'{inout_folder}{file}.jpg'
print(f"Loading mask from {image_path}")
mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
height, width = mask.shape

# IMPORTANT: Adjust this cutoff value to completely sever the trees from the base! # Try increasing this slightly if they are still connected
print(f"Image dimensions: {width}x{height}, Cutoff value: {cutoff_value}")
mask[height-cutoff_value:height, :] = 0 

# 2. Find contours using CHAIN_APPROX_NONE for maximum detail
# This captures EVERY boundary pixel, creating a very dense, accurate polygon
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

class_id = 0 

# 3. Write to a YOLO formatted text file
with open(f'{output_folder}{file}.txt', 'w') as f:
    for contour in contours:
        # Filter out tiny white specks/noise 
        if cv2.contourArea(contour) < 50:
            continue
            
        # Normalize coordinates between 0.0 and 1.0
        normalized_coords = []
        for point in contour:
            x = point[0][0] / width
            y = point[0][1] / height
            normalized_coords.extend([f"{x:.6f}", f"{y:.6f}"])
            
        # Format the line and write to file
        yolo_line = f"{class_id} " + " ".join(normalized_coords) + "\n"
        f.write(yolo_line)

print("Strict YOLO annotation file generated successfully!")