import os

# Path to dataset labels
dataset_path = r"E:\1_Work_Files\13_Project - DroneAID\Datasets\compiled_dataset"
splits = ['train', 'val', 'test']

for split in splits:
    labels_dir = os.path.join(dataset_path, split, 'labels')
    if not os.path.exists(labels_dir):
        continue

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            # Convert class id from float to int, keep bounding boxes unchanged
            class_id = str(int(float(parts[0])))
            rest = parts[1:]
            new_lines.append(' '.join([class_id] + rest))

        # Overwrite the label file
        with open(label_path, 'w') as f:
            f.write('\n'.join(new_lines))

print("All label class IDs converted to integers successfully.")
