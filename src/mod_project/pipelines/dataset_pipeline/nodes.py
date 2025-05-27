import os
import random
import numpy as np
from PIL import Image
from pathlib import Path
import docker


def load_annotated_data(raw_dataset_images, raw_dataset_labels, class_names):
    paired_data = []

    for image_filename, image in raw_dataset_images.items():
        text_filename = os.path.splitext(image_filename)[0] + ".txt"

        if text_filename not in raw_dataset_labels:
            continue

        labels = raw_dataset_labels[text_filename]
        labels_txt = labels()
        labels_list = []

        for line in labels_txt.strip().splitlines():
            cls_id, cx, cy, bw, bh = map(float, line.split())
            labels_list.append([int(cls_id), cx, cy, bw, bh])

        paired_data.append({
            "image_filename": image_filename,
            "image": image,
            "labels_filename": text_filename,
            "labels": labels_list
        })

    return paired_data


def preprocess_for_lama(paired_dataset, num_images):
    images = {}
    labels = {}

    selected = random.sample(paired_dataset, num_images)

    for data in selected:
        image = data["image"]()
        w, h = image.size

        mask = np.zeros((h, w), dtype=np.uint8)
        label_to_remove = random.choice(data["labels"])

        cls_id, cx, cy, bw, bh = label_to_remove
        x1 = int((cx - bw / 2) * w)
        x2 = int((cx + bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        y2 = int((cy + bh / 2) * h)
        mask[y1:y2, x1:x2] = 255

        remaining_lines = [
            f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            for cls_id, cx, cy, bw, bh in data["labels"]
            if [cls_id, cx, cy, bw, bh] != label_to_remove
        ]

        base_name = os.path.splitext(data["image_filename"])[0]
        images[base_name + ".png"] = data["image"]
        images[base_name + "_mask001.png"] = Image.fromarray(mask)
        labels[base_name + ".txt"] = "\n".join(remaining_lines)

    return images, labels


def simulate_missing_parts(lama_input_images):
    import tempfile
    input_dir = Path("data/05_model_input/images").resolve()
    model_dir = Path("data/06_models/lama/big-lama").resolve()
    output_dir = Path(tempfile.mkdtemp())

    client = docker.from_env()

    client.containers.run(
        image="windj007/lama",
        command=[
            "python3", "/home/user/project/bin/predict.py",
            f"model.path=/data/checkpoint",
            f"indir=/data/input",
            f"outdir=/data/output",
            "dataset.img_suffix=.png"
        ],
        volumes={
            str(Path("data/06_models/lama").resolve()): {'bind': '/home/user/project', 'mode': 'ro'},
            str(model_dir): {'bind': '/data/checkpoint', 'mode': 'ro'},
            str(input_dir): {'bind': '/data/input', 'mode': 'ro'},
            str(output_dir): {'bind': '/data/output', 'mode': 'rw'},
        },
        user=f"{os.getuid()}:{os.getgid()}",
        runtime="nvidia",
        remove=True,
        detach=False,
        tty=True
    )

    images = {}
    for image_path in output_dir.glob("*.png"):
        image = Image.open(image_path)
        images[image_path.name] = image

    return images


def select_normal_images(paired_dataset, inpainted_labels, num_images):
    images = {}
    labels = {}

    filtered_dataset = [
        entry for entry in paired_dataset
        if entry["labels_filename"] not in inpainted_labels
    ]

    selected = random.sample(filtered_dataset, num_images)

    for data in selected:
        base_name = os.path.splitext(data["image_filename"])[0]
        images[base_name + ".png"] = data["image"]
        labels[base_name + ".txt"] = "\n".join([
            f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            for cls_id, cx, cy, bw, bh in data["labels"]
        ])

    return images, labels
