#!/usr/bin/env python3

import os
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import sounddevice as sd

# Costanti
WEIGHTS_DIR = 'weights'  # Directory contenente i file dei pesi
COMMANDS_LIST_FILE = 'commands_list.txt'
SAMPLE_RATE = 16000  # Frequenza di campionamento utilizzata per MFCC
DURATION = 1  # Durata della registrazione in secondi
CONFIDENCE_THRESHOLD = 0.7  # Soglia di confidenza per considerare la parola rilevata

# Carica le classi dal file salvato durante il training
def load_keywords(filepath):
  with open(filepath, 'r') as f:
    return [line.strip() for line in f]

keywords = load_keywords(COMMANDS_LIST_FILE)

# Funzione per preprocessare l'audio (durata fissa)
def preprocess_audio(waveform, sample_rate):
  # Durata fissa di 1 secondo
  fixed_length = DURATION  # in secondi
  num_samples = int(fixed_length * sample_rate)

  if waveform.size(1) > num_samples:
    waveform = waveform[:, :num_samples]
  else:
    padding = num_samples - waveform.size(1)
    waveform = torch.nn.functional.pad(waveform, (0, padding))

  return waveform

# Funzione per l'estrazione delle caratteristiche (MFCC)
def extract_features(waveform, sample_rate=SAMPLE_RATE):
  mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=sample_rate,
    n_mfcc=40,
    melkwargs={
      'n_fft': 1024,
      'hop_length': 512,
      'n_mels': 40,
    }
  )
  mfcc = mfcc_transform(waveform)
  return mfcc

# Definizione del modello MLP
class MLP(nn.Module):
  def __init__(self, input_size, hidden_sizes, num_classes):
    super(MLP, self).__init__()
    layers = []
    in_size = input_size

    for h in hidden_sizes:
      layers.append(nn.Linear(in_size, h))
      layers.append(nn.ReLU())
      in_size = h

    layers.append(nn.Linear(in_size, num_classes))
    layers.append(nn.LogSoftmax(dim=1))  # Per classificazione multi-classe

    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

# Funzione per caricare i pesi dai file .txt
def load_weights_from_txt(model, weights_dir):
  for name, param in model.named_parameters():
    # Genera il nome del file basato sul nome del parametro
    filename = f"{name.replace('.', '_')}.txt"
    filepath = os.path.join(weights_dir, filename)
    if os.path.exists(filepath):
      try:
        # Carica i pesi dal file
        weights = np.loadtxt(filepath, delimiter=',')
        # Rimodella i pesi per adattarsi alla forma del parametro
        weights = weights.reshape(param.shape)
        # Assegna i pesi al parametro
        param.data = torch.from_numpy(weights).float()
        print(f"Loaded weights for '{name}' from '{filepath}'")
      except Exception as e:
        print(f"Error loading weights for '{name}' from '{filepath}': {e}")
    else:
      print(f"File '{filepath}' not found. Unable to load weights for '{name}'.")

# Inizializzazione del modello
def initialize_model(hidden_sizes, num_classes):
  # Crea un input di esempio per determinare la dimensione dell'input
  dummy_waveform = torch.zeros(1, SAMPLE_RATE * DURATION)
  dummy_features = extract_features(dummy_waveform)
  input_size = dummy_features.numel()
  model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, num_classes=num_classes)
  return model

# Funzione principale di inferenza live
def run_live_inference():
  # Inizializzazione del modello
  model = initialize_model(hidden_sizes=[128, 64], num_classes=len(keywords))

  # Carica i pesi dal file .txt
  load_weights_from_txt(model, WEIGHTS_DIR)

  # Imposta il modello in modalità valutazione
  model.eval()

  # Mappatura dagli indici alle classi
  idx_to_class = {idx: word for idx, word in enumerate(keywords)}

  print("Listening... Press Ctrl+C to stop.")

  try:
    while True:
      # Registra audio dal microfono
      # print("Please speak a keyword...")

      audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
      sd.wait()  # Attende che la registrazione sia terminata
      waveform = torch.from_numpy(audio.T)

      # Preprocessa l'audio
      waveform = preprocess_audio(waveform, SAMPLE_RATE)

      # Estrai le caratteristiche
      features = extract_features(waveform)
      features = features.unsqueeze(0)  # Aggiungi la dimensione del batch

      # Appiattisci l'input come richiesto dal modello
      inputs = features.reshape(features.size(0), -1)

      # Esegui l'inferenza
      with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.exp(outputs)  # Converti log-probabilità in probabilità
        confidence, predicted = torch.max(probabilities, 1)
        predicted_label = idx_to_class[predicted.item()]
        confidence_score = confidence.item()

      # Verifica se la confidenza supera la soglia
      if confidence_score >= CONFIDENCE_THRESHOLD:
        print(f"Predicted Keyword: {predicted_label} (Confidence: {confidence_score*100:.2f}%)\n")
      # else:
        # Nessuna parola rilevata con sufficiente confidenza
        # print("No keyword detected.\n")

  except KeyboardInterrupt:
    print("\nInference stopped.")

if __name__ == '__main__':
  run_live_inference()

