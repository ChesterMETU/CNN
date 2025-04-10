import os

def capitalize_folders(root_dir):
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path):
            capitalized_name = folder_name.capitalize()
            capitalized_path = os.path.join(root_dir, capitalized_name)

            if folder_name != capitalized_name:
                os.rename(folder_path, capitalized_path)
                print(f"✅ Renamed: {folder_name} → {capitalized_name}")

# Apply to training and test sets
capitalize_folders('Dataset/Training')
capitalize_folders('Dataset/Test')
