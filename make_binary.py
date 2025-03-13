import os

def update_labels(folder_path):
    labels_path = os.path.join(folder_path, "labels")
    
    if not os.path.exists(labels_path):
        print(f"Labels folder not found: {labels_path}")
        return
    
    for filename in os.listdir(labels_path):
        file_path = os.path.join(labels_path, filename)
        
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                lines = file.readlines()
            
            if lines:
                with open(file_path, "w") as file:
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            parts[0] = "0"  # Change the first digit to 0
                            file.write(" ".join(parts) + "\n")
            
            print(f"Updated: {file_path}")
        else:
            print(f"Skipped: {file_path} (Not a file)")

# Paths to train, test, and valid folders
base_path = "."
folders = ["train", "test", "valid"]

for folder in folders:
    update_labels(os.path.join(base_path, folder))

print("Label update complete.")
