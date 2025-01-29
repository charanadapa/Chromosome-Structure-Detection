#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile

with zipfile.ZipFile('orginalmy.zip', 'r') as zip_ref:
    zip_ref.extractall()


# In[3]:


import os
from collections import Counter

# Define the path to your dataset
dataset_path = "C:/Users/HOME/Desktop/orginalmy"

# Get a list of all class folders
classes = os.listdir(dataset_path)

# Initialize a Counter to count images in each class
image_count = Counter()

# Loop through each class folder
for cls in classes:
    class_folder = os.path.join(dataset_path, cls)
    if os.path.isdir(class_folder):
        # Count the number of images in this class folder
        num_images = len(os.listdir(class_folder))
        image_count[cls] = num_images

# Print the number of classes and the number of images per class
print(f"Number of classes: {len(classes)}")
for cls, count in image_count.items():
    print(f"Class '{cls}': {count} images")


# # data cleaning 

# In[10]:


import os
from PIL import Image  # Ensure this import is at the top


# In[11]:


def clean_image_dataset(dataset_path):
    # Define valid image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    
    # Loop through each class directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check if the file has a valid image extension
            if not file.lower().endswith(valid_extensions):
                print(f"Removing non-image file: {file_path}")
                os.remove(file_path)
                continue
            
            # Try opening the image to detect corruption
            try:
                img = Image.open(file_path)
                img.verify()  # Check if the image is corrupted
            except (IOError, SyntaxError) as e:
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)
        
        # Remove empty directories
        if not os.listdir(root):
            print(f"Removing empty directory: {root}")
            os.rmdir(root)

# Define the path to your dataset
dataset_path = "C:/Users/HOME/Desktop/orginalmy"

# Clean the dataset
clean_image_dataset(dataset_path)

print("Data cleaning complete.")


# # data agumentation

# In[4]:


import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# Define paths
dataset_path = "C:/Users/HOME/Desktop/orginalmy"
output_path = '/path/to/your/output_dataset'  # Where to save the augmented images

# Create directories if they don't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

abnormal_path = os.path.join(dataset_path, 'abnormal')
normal_path = os.path.join(dataset_path, 'normal')

output_abnormal_path = os.path.join(output_path, 'abnormal')
output_normal_path = os.path.join(output_path, 'normal')

os.makedirs(output_abnormal_path, exist_ok=True)
os.makedirs(output_normal_path, exist_ok=True)

# Initialize the ImageDataGenerator with your desired augmentations
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to augment images
def augment_images(class_path, output_class_path, target_count):
    images = os.listdir(class_path)
    current_count = len(images)
    
    # Save original images to the output directory
    for img_name in images:
        shutil.copy(os.path.join(class_path, img_name), output_class_path)
    
    # Calculate how many images need to be generated
    needed_images = target_count - current_count
    
    i = 0
    while i < needed_images:
        img_name = np.random.choice(images)
        img = load_img(os.path.join(class_path, img_name))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        # Generate new images
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_class_path, save_prefix='aug', save_format='jpg'):
            i += 1
            if i >= needed_images:
                break

# Augment the abnormal class to 857 images
augment_images(abnormal_path, output_abnormal_path, 857)

# Augment the normal class to 945 images
augment_images(normal_path, output_normal_path, 945)

print("Augmentation complete.")


# In[8]:


output_path = output_path = "C:/Users/HOME/Desktop/agumentedmy"

