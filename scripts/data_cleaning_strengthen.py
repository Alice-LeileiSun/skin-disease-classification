#data_cleaning_strengthen.py

import os
import shutil
from sklearn.model_selection import train_test_split

# 原始数据路径
data_dir = 'data/raw/'
output_dir = 'data/'

# Helper function to ensure directory exists
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Process each category
categories = os.listdir(data_dir)
for category in categories:
    category_path = os.path.join(data_dir, category)
    
    # Skip if not a directory
    if not os.path.isdir(category_path):
        continue

    # Get all image files
    images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
    if len(images) == 0:
        print(f"Skipping category '{category}' as it contains no image files.")
        continue

    if len(images) < 2:
        print(f"Category '{category}' has fewer than 2 images. Assigning all to test set.")
        ensure_dir(os.path.join(output_dir, 'test', category))
        for file in images:
            shutil.move(os.path.join(category_path, file), os.path.join(output_dir, 'test', category, file))
        continue

    # Split dataset
    train, test = train_test_split(images, test_size=0.2, random_state=42)
    
    # Handle cases where test set has fewer than 2 images
    if len(test) < 2:
        print(f"Category '{category}' - Test set too small, skipping validation split.")
        val = []  # No validation set
    else:
        val, test = train_test_split(test, test_size=0.5, random_state=42)

    # Create directories
    ensure_dir(os.path.join(output_dir, 'train', category))
    ensure_dir(os.path.join(output_dir, 'val', category))
    ensure_dir(os.path.join(output_dir, 'test', category))

    # Move files
    for file in train:
        shutil.move(os.path.join(category_path, file), os.path.join(output_dir, 'train', category, file))
    for file in val:
        shutil.move(os.path.join(category_path, file), os.path.join(output_dir, 'val', category, file))
    for file in test:
        shutil.move(os.path.join(category_path, file), os.path.join(output_dir, 'test', category, file))

    # Log summary
    print(f"Category '{category}' - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

print('Data split and organization complete!')
