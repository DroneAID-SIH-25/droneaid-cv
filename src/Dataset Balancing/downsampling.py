import os
import random
import shutil

# --- CONFIG ---
dataset_path = r"E:\1_Work_Files\13_Project - DroneAID\Datasets\compiled_dataset"
split = "train"  # Only downsample train split
classes = ['person', 'fire', 'smoke', 'small_vehicle', 'large_vehicle', 'two_wheeler']

# Target maximum instances per class
target_instances = {
    'person': 5000,
    'fire': 3000,
    'smoke': 2500
}

images_dir = os.path.join(dataset_path, split, 'images')
labels_dir = os.path.join(dataset_path, split, 'labels')

# --- FUNCTIONS ---
def get_class_instances(labels_dir):
    """Returns dictionary: class -> list of label files containing that class"""
    instances = {cls: [] for cls in classes}
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            image_classes = set()
            for line in lines:
                class_id = int(float(line.strip().split()[0]))
                image_classes.add(classes[class_id])
            for cls in image_classes:
                instances[cls].append(label_file)
    return instances

# --- DOWNsampling ---
instances = get_class_instances(labels_dir)

for cls, target in target_instances.items():
    label_files = instances[cls]
    current_count = len(label_files)
    print(f"{split} - {cls}: current images = {current_count}, target = {target}")

    if current_count <= target:
        print(f"{cls}: no downsampling needed")
        continue

    # Randomly select files to delete
    to_delete = random.sample(label_files, current_count - target)
    
    for label_file in to_delete:
        base_name = os.path.splitext(label_file)[0]
        img_file_jpg = os.path.join(images_dir, base_name + '.jpg')
        img_file_png = os.path.join(images_dir, base_name + '.png')
        label_path = os.path.join(labels_dir, label_file)
        
        # Delete label
        if os.path.exists(label_path):
            os.remove(label_path)
        # Delete image (check jpg or png)
        if os.path.exists(img_file_jpg):
            os.remove(img_file_jpg)
        elif os.path.exists(img_file_png):
            os.remove(img_file_png)

    print(f"{cls}: downsampled to {target} images")
