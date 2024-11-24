#!/usr/bin/env python3

import os
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import pyaudio
import wave

# Suppress ALSA warnings
import warnings
warnings.filterwarnings("ignore")

# Carica le classi dal file salvato durante il training
with open('commands_list.txt', 'r') as f:
  keywords = [line.strip() for line in f]

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
def load_weights_from_txt(model):
  for name, param in model.named_parameters():
    filename = f"{name.replace('.', '_')}.txt"
    if os.path.exists(filename):
      weights = np.loadtxt(filename, delimiter=',')
      weights = weights.reshape(param.shape)
      param.data = torch.from_numpy(weights).float()
      print(f"Loaded weights for '{name}' from '{filename}'")
    else:
      print(f"File '{filename}' not found. Unable to load weights for '{name}'.")

# Funzione per registrare audio dal microfono
def record_audio(filename, duration=1):
  CHUNK = 1024
  FORMAT = pyaudio.paInt16
  CHANNELS = 1
  RATE = 44100  # Sample rate di registrazione
  RECORD_SECONDS = duration
  WAVE_OUTPUT_FILENAME = filename

  p = pyaudio.PyAudio()

  # Controlla se esiste un dispositivo di input
  if p.get_device_count() == 0:
    print("Nessun dispositivo di input audio trovato.")
    return False

  stream = p.open(format=FORMAT,
          channels=CHANNELS,
          rate=RATE,
          input=True,
          frames_per_buffer=CHUNK)

  print("Recording...")

  frames = []

  for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

  print("Finished recording.")

  # Ferma e chiudi lo stream
  stream.stop_stream()
  stream.close()
  p.terminate()

  # Salva l'audio registrato in un file WAV
  wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(RATE)
  wf.writeframes(b''.join(frames))
  wf.close()
  return True

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
def extract_features(waveform):
  mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
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
dummy_waveform = torch.zeros(1, 16000)  # 1 secondo di audio a 16kHz
dummy_features = extract_features(dummy_waveform)
input_size = dummy_features.numel()

num_classes = len(keywords)
hidden_sizes = [128, 64]

model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, num_classes=num_classes)

# Carica i pesi dal file .txt
load_weights_from_txt(model)

# Imposta il modello in modalità valutazione
model.eval()

# Mappatura dagli indici alle classi
idx_to_class = {idx: word for idx, word in enumerate(keywords)}

# Richiedi all'utente di premere Invio per iniziare la registrazione
input("Premi Invio per iniziare la registrazione...")

# Permetti all'utente di impostare la durata della registrazione
try:
  duration = float(input("Inserisci la durata della registrazione in secondi (default 1): ") or 1)
except ValueError:
  print("Input non valido. Utilizzo la durata di default di 1 secondo.")
  duration = 1.0

# Registra audio dal microfono
audio_filename = 'recorded_audio.wav'
success = record_audio(audio_filename, duration=duration)

if not success:
  print("Errore nella registrazione audio.")
  exit()

# Carica l'audio registrato
waveform, sample_rate = torchaudio.load(audio_filename)

# Resampling se necessario
if sample_rate != 16000:
  resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
  waveform = resampler(waveform)
  sample_rate = 16000

# Preprocessa l'audio
waveform = preprocess_audio(waveform, sample_rate)

# Estrai le caratteristiche
features = extract_features(waveform)

# Appiattisci le caratteristiche per adattarle al modello
inputs = features.reshape(1, -1)

# Esegui l'inferenza
with torch.no_grad():
  outputs = model(inputs)
  _, predicted = torch.max(outputs.data, 1)
  predicted_label = idx_to_class[predicted.item()]
  print(f"Parola chiave predetta: {predicted_label}")

print("Programma terminato.")

