import cv2
import numpy as np

# 1. Load your binary mask
image_path = 'Easy.jpg'
mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
height, width = mask.shape

# IMPORTANT: Black out the bottom solid substrate so it doesn't get labeled as a dendrite.
# You will need to tweak the '150' to perfectly match the height of the substrate line.
mask[height-197:height, :] = 0 

# 2. Find the contours (outlines) of the white shapes
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

class_id = 0 # Assuming 'dendrite' is your first/only class

# 3. Write to a YOLO formatted text file
with open(f'{image_path.split(".")[0]}.txt', 'w') as f:
    for contour in contours:
        # Filter out tiny white specks/noise (adjust the 50 as needed)
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

print("YOLO annotation file generated successfully!")