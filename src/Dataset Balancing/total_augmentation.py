import os
import random
import cv2
import albumentations as A

# --- CONFIG ---
dataset_path = r"E:\1_Work_Files\13_Project - DroneAID\Datasets\compiled_dataset"
split = "train"  # Only augment train split
classes = ['person', 'fire', 'smoke', 'small_vehicle', 'large_vehicle', 'two_wheeler']

# Target total instances per class
target_instances = {
    'person': 5000,
    'fire': 3000,
    'smoke': 2500,
    'small_vehicle': 2500,
    'large_vehicle': 2500,
    'two_wheeler': 2500
}

max_attempts_per_class = 5000  # to avoid infinite loops

# Albumentations augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# --- FUNCTIONS ---
def get_instances(labels_dir):
    """Returns dict: class -> list of (label_file, list of class_ids)"""
    instances = {cls: [] for cls in classes}
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        path = os.path.join(labels_dir, label_file)
        with open(path, 'r') as f:
            lines = f.readlines()
        class_ids = [int(float(line.strip().split()[0])) for line in lines]
        image_classes = set(classes[cid] for cid in class_ids)
        for cls in image_classes:
            instances[cls].append((label_file, class_ids))
    return instances

def count_total_instances(instances_dict):
    """Counts total number of instances per class"""
    counts = {cls: 0 for cls in classes}
    for cls, items in instances_dict.items():
        for _, class_ids in items:
            counts[cls] += sum(1 for cid in class_ids if classes[cid]==cls)
    return counts

# --- MAIN UPSAMPLING ---
images_dir = os.path.join(dataset_path, split, 'images')
labels_dir = os.path.join(dataset_path, split, 'labels')

instances_dict = get_instances(labels_dir)
current_counts = count_total_instances(instances_dict)
print("Initial instance counts:", current_counts)

for cls in classes:
    target = target_instances[cls]
    current = current_counts[cls]
    print(f"Processing {cls}: current instances = {current}, target = {target}")

    if current >= target:
        print(f"{cls}: already at or above target, skipping")
        continue

    attempts = 0
    while current < target and attempts < max_attempts_per_class:
        attempts += 1
        if not instances_dict[cls]:
            print(f"No source images for {cls}, cannot augment further")
            break

        # pick a random image
        label_file, class_ids = random.choice(instances_dict[cls])
        base_name = os.path.splitext(label_file)[0]
        img_path_jpg = os.path.join(images_dir, base_name + '.jpg')
        img_path_png = os.path.join(images_dir, base_name + '.png')
        img_path = img_path_jpg if os.path.exists(img_path_jpg) else img_path_png
        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        # bounding boxes for augmentation
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
        bboxes = []
        bboxes_class_ids = []
        for line in lines:
            parts = line.strip().split()
            cid = int(float(parts[0]))
            if classes[cid] != cls:
                continue
            bbox = list(map(float, parts[1:]))
            bboxes.append(bbox)
            bboxes_class_ids.append(cid)

        if not bboxes:
            continue

        # augment
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=bboxes_class_ids)
        except:
            continue

        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_class_ids = augmented['class_labels']

        if not aug_bboxes:
            continue

        # save
        new_index = len(os.listdir(images_dir)) + 1
        new_img_name = f"{new_index}.jpg"
        new_lbl_name = f"{new_index}.txt"
        cv2.imwrite(os.path.join(images_dir, new_img_name), aug_image)
        with open(os.path.join(labels_dir, new_lbl_name), 'w') as f:
            for cid, bbox in zip(aug_class_ids, aug_bboxes):
                f.write(f"{cid} {' '.join(map(str, bbox))}\n")

        current += len(aug_class_ids)

    print(f"{cls}: final instance count = {current}")
