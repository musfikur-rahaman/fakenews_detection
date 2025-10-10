import shutil
import kagglehub
import os
import sys
import zipfile
# Download dataset
path = kagglehub.dataset_download("bhavikjikadara/fake-news-detection")

# Destination: current working directory
destination = os.getcwd()

# Copy all contents from cache to current folder
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(destination, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print(f"Copied dataset files to: {destination}")
