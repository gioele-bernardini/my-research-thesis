#!/usr/bin/env python3.11

import os
import urllib.request
import tarfile

# Google Speech Commands dataset URL (version 2)
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATASET_DIR = "speech_data"
ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"

# Create the dataset directory if it doesn't exist
os.makedirs(DATASET_DIR, exist_ok=True)

# Full path for the downloaded archive
archive_path = os.path.join(DATASET_DIR, ARCHIVE_NAME)

# Download the dataset if it's not already present
if not os.path.exists(archive_path):
    print(f"Downloading dataset from {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, archive_path)
    print("Download completed!")

# Extract the dataset
print("Extracting files...")
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(DATASET_DIR)

print(f"Dataset successfully extracted to '{DATASET_DIR}'!")

# Optional: Remove the archive to save space
os.remove(archive_path)
print("Archive removed.")
