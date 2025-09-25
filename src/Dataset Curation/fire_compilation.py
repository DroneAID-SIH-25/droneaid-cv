import os
import shutil

# Source dataset
source_dataset = r"E:\1_Work_Files\13_Project - DroneAID\Datasets\fire"

# Target compiled dataset
target_dataset = r"E:\1_Work_Files\13_Project - DroneAID\Datasets\compiled_dataset"

# Original classes in this dataset
original_classes = ['fire', 'smoke']

# Mapping to new unified classes
class_mapping = {
    'fire': 1,
    'smoke': 2
}

# Create the folder structure in compiled_dataset
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(target_dataset, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_dataset, split, 'labels'), exist_ok=True)

# Function to process each split
def process_split(split):
    src_images = os.path.join(source_dataset, split, 'images')
    src_labels = os.path.join(source_dataset, split, 'labels')
    
    # Skip if labels folder doesn't exist
    if not os.path.exists(src_labels):
        print(f"Skipping {split}: labels folder not found.")
        return
    
    tgt_images = os.path.join(target_dataset, split, 'images')
    tgt_labels = os.path.join(target_dataset, split, 'labels')
    
    for label_file in os.listdir(src_labels):
        if label_file.endswith('.txt'):
            src_label_path = os.path.join(src_labels, label_file)
            tgt_label_path = os.path.join(tgt_labels, label_file)
            
            # Read original label
            with open(src_label_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    orig_class_id = int(parts[0])
                    orig_class_name = original_classes[orig_class_id]
                    if orig_class_name in class_mapping:
                        new_class_id = class_mapping[orig_class_name]
                        new_line = f"{new_class_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}"
                        new_lines.append(new_line)
            
            # Write new label file
            with open(tgt_label_path, 'w') as f:
                f.write("\n".join(new_lines))
            
            # Copy corresponding image
            image_name = os.path.splitext(label_file)[0] + '.jpg'
            src_image_path = os.path.join(src_images, image_name)
            tgt_image_path = os.path.join(tgt_images, image_name)
            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, tgt_image_path)

# Process train, val, test
for split in ['train', 'val', 'test']:
    process_split(split)

print("Fire dataset successfully merged into compiled_dataset!")
