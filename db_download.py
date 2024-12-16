#!/usr/bin/env python3

import os
import urllib.request
import tarfile

# URL del dataset di Google Speech Commands
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# Nome del file compresso
ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"
# Nome della cartella di estrazione
OUTPUT_DIR = "speech-commands"

def download_and_extract(url=DATA_URL, archive_name=ARCHIVE_NAME, output_dir=OUTPUT_DIR):
  # Crea la cartella di output se non esiste
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Scarica il file solo se non esiste già
  if not os.path.exists(archive_name):
    print(f"Scaricamento del dataset da {url}...")
    urllib.request.urlretrieve(url, archive_name)
    print("Download completato.")
  else:
    print("Il file è già presente, non è necessario scaricarlo.")

  # Estrai il contenuto all'interno della cartella di output
  print("Estrazione del dataset...")
  with tarfile.open(archive_name, "r:gz") as tar:
    tar.extractall(path=output_dir)
  print(f"Estrazione completata nella cartella '{output_dir}'.")

if __name__ == "__main__":
  download_and_extract()
