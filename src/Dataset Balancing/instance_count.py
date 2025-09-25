import os

# Path to the compiled dataset
compiled_dataset = r"E:\1_Work_Files\13_Project - DroneAID\Datasets\compiled_dataset"

# Unified class names
classes = ['person', 'fire', 'smoke', 'small_vehicle', 'large_vehicle', 'two_wheeler']
class_counts = {cls: 0 for cls in classes}

# Function to count instances in a folder
def count_instances_in_split(split):
    labels_dir = os.path.join(compiled_dataset, split, 'labels')
    if not os.path.exists(labels_dir):
        print(f"Skipping {split}: labels folder not found.")
        return
    
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        if class_id < len(classes):
                            class_counts[classes[class_id]] += 1

# Count instances in train, val, test
for split in ['train', 'val', 'test']:
    count_instances_in_split(split)

# Print results
print("Total instances of each class in compiled dataset:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")
