#!/usr/bin/env python3

import os
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import pyaudio
import wave
import warnings

# Suppress ALSA warnings
warnings.filterwarnings("ignore")

# Costanti
WEIGHTS_DIR = 'weights'  # Directory contenente i file dei pesi
COMMANDS_LIST_FILE = 'commands_list.txt'
AUDIO_OUTPUT_FILE = 'recorded_audio.wav'
DEFAULT_DURATION = 1  # Durata di default della registrazione in secondi
SAMPLE_RATE = 16000  # Frequenza di campionamento per l'elaborazione
CHUNK = 1024  # Dimensione del buffer per PyAudio
FORMAT = pyaudio.paInt16  # Formato di registrazione
CHANNELS = 1  # Numero di canali audio

# Carica le classi dal file salvato durante il training
def load_keywords(filepath):
  try:
    with open(filepath, 'r') as f:
      keywords = [line.strip() for line in f]
    print(f"Loaded {len(keywords)} keywords from '{filepath}'.")
    return keywords
  except FileNotFoundError:
    print(f"File '{filepath}' non trovato.")
    exit(1)

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
    filename = f"{name.replace('.', '_')}.txt"
    filepath = os.path.join(weights_dir, filename)
    if os.path.exists(filepath):
      try:
        weights = np.loadtxt(filepath, delimiter=',')
        weights = weights.reshape(param.shape)
        param.data = torch.from_numpy(weights).float()
        print(f"Loaded weights for '{name}' from '{filepath}'.")
      except Exception as e:
        print(f"Errore nel caricamento dei pesi per '{name}' da '{filepath}': {e}")
    else:
      print(f"File '{filepath}' non trovato. Impossibile caricare i pesi per '{name}'.")

# Funzione per registrare audio dal microfono
def record_audio(filename, duration=DEFAULT_DURATION):
  p = pyaudio.PyAudio()

  # Controlla se esiste un dispositivo di input
  if p.get_device_count() == 0:
    print("Nessun dispositivo di input audio trovato.")
    p.terminate()
    return False

  try:
    stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK)
  except Exception as e:
    print(f"Errore nell'apertura dello stream audio: {e}")
    p.terminate()
    return False

  print("Registrazione in corso...")

  frames = []

  try:
    for _ in range(0, int(SAMPLE_RATE / CHUNK * duration)):
      data = stream.read(CHUNK, exception_on_overflow=False)
      frames.append(data)
  except Exception as e:
    print(f"Errore durante la registrazione: {e}")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return False

  print("Registrazione terminata.")

  # Ferma e chiudi lo stream
  stream.stop_stream()
  stream.close()
  p.terminate()

  # Salva l'audio registrato in un file WAV
  try:
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio salvato in '{filename}'.")
    return True
  except Exception as e:
    print(f"Errore nel salvataggio del file audio: {e}")
    return False

# Funzione per preprocessare l'audio
def preprocess_audio(waveform, sample_rate):
  fixed_length = 1  # Durata fissa di 1 secondo
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

# Inizializzazione del modello
def initialize_model(hidden_sizes, num_classes):
  dummy_waveform = torch.zeros(1, SAMPLE_RATE)  # 1 secondo di audio a SAMPLE_RATE
  dummy_features = extract_features(dummy_waveform)
  input_size = dummy_features.numel()
  model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, num_classes=num_classes)
  print(f"Modello inizializzato con input size {input_size} e {num_classes} classi.")
  return model

# Funzione principale di inferenza
def run_inference():
  # Carica le keyword
  keywords = load_keywords(COMMANDS_LIST_FILE)

  # Inizializza il modello
  hidden_sizes = [128, 64]
  num_classes = len(keywords)
  model = initialize_model(hidden_sizes, num_classes)

  # Carica i pesi
  load_weights_from_txt(model, WEIGHTS_DIR)

  # Imposta il modello in modalità valutazione
  model.eval()

  # Mappatura dagli indici alle classi
  idx_to_class = {idx: word for idx, word in enumerate(keywords)}

  # Richiedi all'utente di premere Invio per iniziare la registrazione
  input("Premi Invio per iniziare la registrazione...")

  # Permetti all'utente di impostare la durata della registrazione
  try:
    duration_input = input(f"Inserisci la durata della registrazione in secondi (default {DEFAULT_DURATION}): ")
    duration = float(duration_input) if duration_input.strip() else DEFAULT_DURATION
  except ValueError:
    print("Input non valido. Utilizzo la durata di default di 1 secondo.")
    duration = DEFAULT_DURATION

  # Registra audio dal microfono
  success = record_audio(AUDIO_OUTPUT_FILE, duration=duration)

  if not success:
    print("Errore nella registrazione audio.")
    exit(1)

  # Carica l'audio registrato
  try:
    waveform, sample_rate = torchaudio.load(AUDIO_OUTPUT_FILE)
    print(f"Audio caricato con sample rate {sample_rate} Hz.")
  except Exception as e:
    print(f"Errore nel caricamento del file audio: {e}")
    exit(1)

  # Resampling se necessario
  if sample_rate != SAMPLE_RATE:
    try:
      resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
      waveform = resampler(waveform)
      sample_rate = SAMPLE_RATE
      print(f"Audio resamplato a {sample_rate} Hz.")
    except Exception as e:
      print(f"Errore nel resampling dell'audio: {e}")
      exit(1)

  # Preprocessa l'audio
  waveform = preprocess_audio(waveform, sample_rate)

  # Estrai le caratteristiche
  features = extract_features(waveform)
  print("Estrazione delle caratteristiche completata.")

  # Appiattisci le caratteristiche per adattarle al modello
  inputs = features.reshape(1, -1)

  # Esegui l'inferenza
  with torch.no_grad():
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predicted_label = idx_to_class.get(predicted.item(), "Sconosciuto")
    print(f"Parola chiave predetta: {predicted_label}")

  print("Programma terminato.")

if __name__ == '__main__':
  run_inference()

