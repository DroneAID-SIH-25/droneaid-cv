import os
import random
import cv2
import albumentations as A

# --- CONFIG ---
dataset_path = r"E:\1_Work_Files\13_Project - DroneAID\Datasets\compiled_dataset"
split = "train"
target_class = "smoke"
target_instances = 2500
max_attempts = 5000  # prevent infinite loop

classes = ['person', 'fire', 'smoke', 'small_vehicle', 'large_vehicle', 'two_wheeler']

# Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Paths
images_dir = os.path.join(dataset_path, split, 'images')
labels_dir = os.path.join(dataset_path, split, 'labels')

# --- FUNCTIONS ---
def get_smoke_instances(labels_dir):
    """Return list of tuples (label_file, class_ids) for images containing smoke"""
    smoke_list = []
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
        class_ids = [int(float(line.strip().split()[0])) for line in lines]
        if 2 in class_ids:  # smoke class index
            smoke_list.append((label_file, class_ids))
    return smoke_list

def count_smoke_instances(smoke_list, labels_dir):
    total = 0
    for label_file, class_ids in smoke_list:
        total += sum(1 for cid in class_ids if cid == 2)
    return total

# --- MAIN UPSAMPLING LOOP ---
smoke_instances = get_smoke_instances(labels_dir)
current_count = count_smoke_instances(smoke_instances, labels_dir)
print(f"Current smoke instances: {current_count}, target: {target_instances}")

attempts = 0
while current_count < target_instances and attempts < max_attempts:
    attempts += 1
    label_file, class_ids = random.choice(smoke_instances)
    base_name = os.path.splitext(label_file)[0]
    img_path_jpg = os.path.join(images_dir, base_name + '.jpg')
    img_path_png = os.path.join(images_dir, base_name + '.png')
    img_path = img_path_jpg if os.path.exists(img_path_jpg) else img_path_png
    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    # Prepare bounding boxes for smoke only
    with open(os.path.join(labels_dir, label_file), 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_ids_list = []
    for line in lines:
        parts = line.strip().split()
        cid = int(float(parts[0]))
        if cid != 2:
            continue
        bbox = list(map(float, parts[1:]))
        bboxes.append(bbox)
        class_ids_list.append(cid)

    if not bboxes:
        continue

    # Augment
    try:
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_ids_list)
    except:
        continue

    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_class_ids = augmented['class_labels']

    if not aug_bboxes:
        continue

    # Save augmented image and label
    new_index = len(os.listdir(images_dir)) + 1
    new_img_name = f"{new_index}.jpg"
    new_lbl_name = f"{new_index}.txt"
    cv2.imwrite(os.path.join(images_dir, new_img_name), aug_image)
    with open(os.path.join(labels_dir, new_lbl_name), 'w') as f:
        for cid, bbox in zip(aug_class_ids, aug_bboxes):
            f.write(f"{cid} {' '.join(map(str, bbox))}\n")

    current_count += len(aug_class_ids)

print(f"Finished. Total smoke instances: {current_count}")
