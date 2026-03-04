import cv2
import numpy as np

# The cutoff value you used in the first script
cutoff_value = 120

# File paths
file = 'easy_2'
ann_input_folder = 'annotations/Easy/'
image_input_folder = 'maked_dataset/Easy/'
output_folder = 'visualizations/Easy/'
image_path = f'{image_input_folder}{file}.jpg'
txt_path = f'{ann_input_folder}{file}.txt'

# Load the image in color so we can draw colored lines on it
img = cv2.imread(image_path)
height, width, _ = img.shape

# Read the YOLO text file
try:
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        
        # Extract coordinates (skip class_id at index 0)
        coords = np.array(parts[1:], dtype=float)
        
        # Convert normalized [x, y, x, y...] back to absolute pixel coordinates
        xs = (coords[0::2] * width).astype(np.int32)
        ys = (coords[1::2] * height).astype(np.int32)
        
        # Reshape for OpenCV
        pts = np.stack((xs, ys), axis=-1).reshape((-1, 1, 2))
        
        # Draw the polygon outline in Green
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

except FileNotFoundError:
    print(f"Could not find {txt_path}. Make sure you ran the first script!")

# Draw a Red horizontal line to show exactly where the cutoff happened
y_cutoff_line = height - cutoff_value
cv2.line(img, (0, y_cutoff_line), (width, y_cutoff_line), (0, 0, 255), 2)

# Save the result
output_path = f'{output_folder}{file}_visualization.png'
cv2.imwrite(output_path, img)
print(f"Saved visualization to {output_path}")