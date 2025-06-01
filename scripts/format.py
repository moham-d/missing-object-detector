import os
import shutil
from pathlib import Path

# Base paths
base_path = Path("data/03_primary/inpainted")
label_source_path = Path("data/02_intermediate/inpainted/labels")

# Target folders
image_target = base_path / "images"
label_target = base_path / "labels"

# Ensure target folders exist
image_target.mkdir(parents=True, exist_ok=True)
label_target.mkdir(parents=True, exist_ok=True)

# Denominations to process
denominations = [1, 2, 5, 10, 50, 100]

for denomination in denominations:
    folder = base_path / str(denomination)
    if not folder.exists():
        print(f"Warning: folder {folder} does not exist, skipping")
        continue

    for file in folder.iterdir():
        if file.is_file():
            # New image name
            new_image_name = f"{denomination}_usd_{file.name}"
            new_image_path = image_target / new_image_name

            # Copy image
            print(f"Copying image: {file} → {new_image_path}")
            shutil.copy2(str(file), str(new_image_path))

            # Corresponding label filename
            label_filename = os.path.splitext(file.name)[0] + ".txt"
            label_file_path = label_source_path / label_filename.replace("_mask001", "")

            if label_file_path.exists():
                new_label_name = f"{denomination}_usd_{label_filename}"
                new_label_path = label_target / new_label_name

                # Copy label
                print(f"Copying label: {label_file_path} → {new_label_path}")
                shutil.copy2(str(label_file_path), str(new_label_path))
            else:
                print(f"Warning: Label file {label_file_path} not found for image {file.name}")
