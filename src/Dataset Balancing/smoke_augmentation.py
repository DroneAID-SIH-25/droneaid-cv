import os
import random
import cv2
import albumentations as A

# --- CONFIG ---
dataset_path = r"E:\1_Work_Files\13_Project - DroneAID\Datasets\compiled_dataset"
split = "train"
classes = ['person', 'fire', 'smoke', 'small_vehicle', 'large_vehicle', 'two_wheeler']
smoke_class_index = 2  # smoke

# Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=1.0),  # guaranteed flip
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Paths
images_dir = os.path.join(dataset_path, split, 'images')
labels_dir = os.path.join(dataset_path, split, 'labels')

# Get all images containing smoke
smoke_images = []
for label_file in os.listdir(labels_dir):
    if not label_file.endswith('.txt'):
        continue
    with open(os.path.join(labels_dir, label_file), 'r') as f:
        lines = f.readlines()
    class_ids = [int(float(line.strip().split()[0])) for line in lines]
    if smoke_class_index in class_ids:
        smoke_images.append(label_file)

print(f"Found {len(smoke_images)} images containing smoke.")

# Number of images to generate (double)
num_to_generate = len(smoke_images)
print(f"Will generate {num_to_generate} augmented images.")

# --- AUGMENTATION LOOP ---
for i in range(num_to_generate):
    label_file = random.choice(smoke_images)
    base_name = os.path.splitext(label_file)[0]

    img_path_jpg = os.path.join(images_dir, base_name + '.jpg')
    img_path_png = os.path.join(images_dir, base_name + '.png')
    img_path = img_path_jpg if os.path.exists(img_path_jpg) else img_path_png
    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    # Read all bboxes
    with open(os.path.join(labels_dir, label_file), 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_ids_list = []
    for line in lines:
        parts = line.strip().split()
        cid = int(float(parts[0]))  # convert to int
        bbox = list(map(float, parts[1:]))
        bboxes.append(bbox)
        class_ids_list.append(cid)

    # Apply augmentation
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_ids_list)
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
            f.write(f"{int(cid)} {' '.join(map(str, bbox))}\n")  # force int here

print("Done doubling smoke images! All class IDs are now integers.")
