# check_files.py: Inspect dataset contents and naming conventions
import os

dataset_dir = r"D:\MDM_Project\Dataset"

print(f"Inspecting directory: {dataset_dir}")
if not os.path.exists(dataset_dir):
    print("Error: Directory does not exist.")
else:
    count = 0
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            print(f"Found file: {f}")
            count += 1
            if count >= 15:  # Show first 15 files
                print("... (showing first 15 files)")
                break
        if count >= 15:
            break
    if count == 0:
        print("The directory is completely empty or contains no files.")