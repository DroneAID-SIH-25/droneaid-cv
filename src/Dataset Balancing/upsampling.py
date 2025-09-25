import os
import random
import cv2
import albumentations as A

# --- CONFIG ---
dataset_path = r"E:\1_Work_Files\13_Project - DroneAID\Datasets\compiled_dataset"
splits = ['train']  # Only augment train
classes = ['person', 'fire', 'smoke', 'small_vehicle', 'large_vehicle', 'two_wheeler']

target_instances = {
    'small_vehicle': 2500,
    'large_vehicle': 2500,
    'two_wheeler': 2500
}

max_attempts_per_class = 5000  # Prevent infinite loops

# --- AUGMENTATIONS ---
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# --- FUNCTIONS ---
def get_class_instances(labels_dir):
    """Returns dictionary: class -> list of (image_file, label_lines)"""
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
                instances[cls].append((label_file, lines))
    return instances

# --- UPSAMPLING LOOP ---
for split in splits:
    images_dir = os.path.join(dataset_path, split, 'images')
    labels_dir = os.path.join(dataset_path, split, 'labels')
    
    instances = get_class_instances(labels_dir)
    
    for cls, target in target_instances.items():
        # Count current instances
        current_count = sum([len([line for line in lines if classes[int(float(line.strip().split()[0]))] == cls])
                             for _, lines in instances[cls]])
        print(f"{split} - {cls}: current instances = {current_count}, target = {target}")
        
        attempts = 0
        while current_count < target and attempts < max_attempts_per_class:
            attempts += 1
            # Randomly pick an image
            label_file, label_lines = random.choice(instances[cls])
            img_name = os.path.splitext(label_file)[0] + '.jpg'
            img_path = os.path.join(images_dir, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                # Skip unreadable images
                print(f"Skipping {img_name}: cannot read image file")
                continue
            
            # Prepare bounding boxes
            bboxes = []
            class_labels = []
            for line in label_lines:
                parts = line.strip().split()
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
                bboxes.append([x, y, w, h])
                class_labels.append(class_id)
            
            # Apply augmentation
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                print(f"Augmentation failed for {img_name}: {e}")
                continue
            
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']
            
            # Save augmented image and label
            new_index = len(os.listdir(images_dir)) + 1
            new_image_name = f"{new_index}.jpg"
            new_label_name = f"{new_index}.txt"
            cv2.imwrite(os.path.join(images_dir, new_image_name), aug_image)
            with open(os.path.join(labels_dir, new_label_name), 'w') as f:
                for cid, bbox in zip(aug_labels, aug_bboxes):
                    f.write(f"{int(cid)} {' '.join(map(str, bbox))}\n")
            
            # Update instance count
            current_count += sum(1 for cid in aug_labels if classes[int(cid)] == cls)
        
        if current_count >= target:
            print(f"{cls}: reached target of {target} instances")
        else:
            print(f"{cls}: could not reach target after {max_attempts_per_class} attempts. Current = {current_count}")
