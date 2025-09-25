import os
import random

compiled_dataset = r"E:\1_Work_Files\13_Project - DroneAID\Datasets\compiled_dataset"

def rename_split_random(split):
    images_dir = os.path.join(compiled_dataset, split, 'images')
    labels_dir = os.path.join(compiled_dataset, split, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Skipping {split}: images or labels folder not found.")
        return
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Shuffle images randomly
    random.shuffle(image_files)
    
    temp_map = {}
    
    # Step 1: rename to temporary names to avoid conflicts
    for idx, image_file in enumerate(image_files, start=1):
        old_image_path = os.path.join(images_dir, image_file)
        old_label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')
        
        temp_image_path = os.path.join(images_dir, f"temp_{idx}.jpg")
        temp_label_path = os.path.join(labels_dir, f"temp_{idx}.txt")
        
        os.rename(old_image_path, temp_image_path)
        if os.path.exists(old_label_path):
            os.rename(old_label_path, temp_label_path)
        
        temp_map[idx] = (temp_image_path, temp_label_path)
    
    # Step 2: rename from temporary names to final sequential names
    for idx, (temp_image_path, temp_label_path) in temp_map.items():
        final_image_path = os.path.join(images_dir, f"{idx}.jpg")
        final_label_path = os.path.join(labels_dir, f"{idx}.txt")
        
        os.rename(temp_image_path, final_image_path)
        if os.path.exists(temp_label_path):
            os.rename(temp_label_path, final_label_path)
    
    print(f"{split} renamed randomly: {len(image_files)} files processed.")

# Apply to all splits
for split in ['train', 'val', 'test']:
    rename_split_random(split)

print("All images and labels have been randomly renamed sequentially.")
