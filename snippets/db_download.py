#!/usr/bin/env python3

import os
import urllib.request
import tarfile
import sys

# URL of the dataset
url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# Destination path
data_path = "./speech_commands/"
os.makedirs(data_path, exist_ok=True)

# Name of the downloaded file
filename = "speech_commands_v0.02.tar.gz"

# Progress bar function
def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percentage = downloaded / total_size * 100
    progress = int(50 * downloaded / total_size)
    bar = '[' + '#' * progress + '-' * (50 - progress) + ']'
    sys.stdout.write(f"\rDownloading {bar} {percentage:.2f}%")
    sys.stdout.flush()

# Download the dataset with progress bar
print("Downloading Speech Commands Dataset...")
urllib.request.urlretrieve(url, filename, reporthook=progress_hook)
print("\nDownload completed.")

# Extract the compressed file
print("Extracting files...")
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall(path=data_path)
print("Extraction completed.")

# Remove the compressed file
os.remove(filename)
print("Compressed file removed.")

