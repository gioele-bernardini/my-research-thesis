#!/usr/bin/env python3

import os
import urllib.request
import tarfile

# URL del dataset
url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# Percorso di destinazione
data_path = "./speech_commands/"
os.makedirs(data_path, exist_ok=True)

# Nome del file scaricato
filename = "speech_commands_v0.02.tar.gz"

# Scarica il dataset
print("Downloading Speech Commands Dataset...")
urllib.request.urlretrieve(url, filename)
print("Download completato.")

# Estrai il file
print("Extracting files...")
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall(path=data_path)
print("Estrazione completata.")

# Rimuovi il file compresso
os.remove(filename)
print("File compresso rimosso.")

