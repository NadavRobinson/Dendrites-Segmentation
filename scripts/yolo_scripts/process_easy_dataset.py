import cv2
import numpy as np
import os

# Configuration
input_folder = 'maked_dataset/Easy/'
annotation_output_folder = 'annotations/Easy/'
visualization_output_folder = 'visualizations/Easy/'
cutoffs_file = 'cut_offs.txt'

# Ensure output directories exist
os.makedirs(annotation_output_folder, exist_ok=True)
os.makedirs(visualization_output_folder, exist_ok=True)

# Read cutoffs from file
if not os.path.exists(cutoffs_file):
    print(f"Error: {cutoffs_file} not found.")
    exit(1)

with open(cutoffs_file, 'r') as f:
    # Read each line, strip whitespace, and convert to int if not empty
    cutoffs = [int(line.strip()) for line in f if line.strip()]

print(f"Found {len(cutoffs)} cutoff values. Starting batch process...")

for i, cutoff_value in enumerate(cutoffs):
    file_number = i + 1
    file_base = f'easy_{file_number}'
    image_path = os.path.join(input_folder, f'{file_base}.jpg')
    
    if not os.path.exists(image_path):
        print(f"Skipping {file_base}: Image not found at {image_path}")
        continue

    print(f"--- Processing {file_base} (Cutoff: {cutoff_value}) ---")
    
    # Load image
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not read image {image_path}")
        continue
        
    height, width = mask.shape
    
    # 1. Annotation Generation
    # Create a copy to apply cutoff without destroying original for visualization
    process_mask = mask.copy()
    process_mask[max(0, height-cutoff_value):height, :] = 0 
    
    # Find contours
    contours, _ = cv2.findContours(process_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    class_id = 0 
    annotation_path = os.path.join(annotation_output_folder, f'{file_base}.txt')
    
    yolo_lines = []
    for contour in contours:
        # Filter out tiny white specks/noise 
        if cv2.contourArea(contour) < 50:
            continue
            
        normalized_coords = []
        for point in contour:
            x = point[0][0] / width
            y = point[0][1] / height
            normalized_coords.extend([f"{x:.6f}", f"{y:.6f}"])
            
        yolo_lines.append(f"{class_id} " + " ".join(normalized_coords) + "\n")
        
    with open(annotation_path, 'w') as f:
        f.writelines(yolo_lines)
    print(f"  [+] Saved annotation: {annotation_path}")
        
    # 2. Visualization
    # Load color version for drawing
    vis_img = cv2.imread(image_path)
    
    # Draw contours from the generated YOLO lines
    for line in yolo_lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        
        coords = np.array(parts[1:], dtype=float)
        xs = (coords[0::2] * width).astype(np.int32)
        ys = (coords[1::2] * height).astype(np.int32)
        pts = np.stack((xs, ys), axis=-1).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw Red horizontal line to show the cutoff
    y_cutoff_line = height - cutoff_value
    cv2.line(vis_img, (0, y_cutoff_line), (width, y_cutoff_line), (0, 0, 255), 2)
    
    output_vis_path = os.path.join(visualization_output_folder, f'{file_base}_visualization.png')
    cv2.imwrite(output_vis_path, vis_img)
    print(f"  [+] Saved visualization: {output_vis_path}")

print("\nBatch processing complete!")
