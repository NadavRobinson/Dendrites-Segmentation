import os
import shutil

source_dir = r"c:\Users\yuval\PycharmProjects\Dendrites-Segmentation\output\classic_hard"
dest_dir = r"c:\Users\yuval\PycharmProjects\Dendrites-Segmentation\output\hard_separated"

# Create destination directory
os.makedirs(dest_dir, exist_ok=True)

# Copy 10_separated.png from each subfolder
for subfolder in os.listdir(source_dir):
    subfolder_path = os.path.join(source_dir, subfolder)
    if os.path.isdir(subfolder_path):
        src_file = os.path.join(subfolder_path, "10_separated.png")
        dst_file = os.path.join(dest_dir, f"{subfolder}.png")
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {subfolder}.png")
        else:
            print(f"Missing: {subfolder}/10_separated.png")

print(f"\nDone! Images saved to: {dest_dir}")
