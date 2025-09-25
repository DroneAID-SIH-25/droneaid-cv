import os

# Compiled dataset path
dataset_path = r"E:\1_Work_Files\13_Project - DroneAID\Datasets\compiled_dataset"

classes = ['person', 'fire', 'smoke', 'small_vehicle', 'large_vehicle', 'two_wheeler']

# Dictionary to store number of images containing each class
images_per_class = {cls: set() for cls in classes}

# Loop through splits
for split in ['train', 'val', 'test']:
    labels_dir = os.path.join(dataset_path, split, 'labels')
    if not os.path.exists(labels_dir):
        continue
    
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            # Keep track of which classes appear in this image
            classes_in_image = set()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    classes_in_image.add(classes[class_id])
            # Add image filename to all relevant class sets
            for cls in classes_in_image:
                images_per_class[cls].add(label_file)

# Print total number of images containing each class
print("Total images per class:")
for cls in classes:
    print(f"{cls}: {len(images_per_class[cls])}")
