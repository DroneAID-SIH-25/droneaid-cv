import os

# Paths
labels_dir = r"E:\1_Work_Files\13_Project - DroneAID\Datasets\vehicle dataset\train\labels"

# Class info
classes = ['car', 'threewheel', 'bus', 'truck', 'motorbike', 'van']
class_counts = {cls: 0 for cls in classes}

# Loop through all label files
for file in os.listdir(labels_dir):
    if file.endswith(".txt"):
        file_path = os.path.join(labels_dir, file)
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])
                    if class_id < len(classes):
                        class_counts[classes[class_id]] += 1

# Print results
print("Class instance counts in training set:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")
