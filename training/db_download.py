#!/usr/bin/env python3

import os
import urllib.request
import tarfile

# URL of the dataset
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"
OUTPUT_DIR = "speech-commands"

def download_and_extract(url=DATA_URL, archive_name=ARCHIVE_NAME, output_dir=OUTPUT_DIR):
    # Create the folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download the file only if it does not exist
    if not os.path.exists(archive_name):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, archive_name)
        print("Download completed.")
    else:
        print("File already present, no need to download.")

    # Extract the content
    print("Extracting dataset...")
    with tarfile.open(archive_name, "r:gz") as tar:
        tar.extractall(path=output_dir)
    print(f"Extraction completed in the '{output_dir}' directory.")

    # Rimuovi l'archivio dopo l'estrazione
    if os.path.exists(archive_name):
        os.remove(archive_name)
        print(f"Archive {archive_name} has been removed.")

if __name__ == "__main__":
    download_and_extract()
