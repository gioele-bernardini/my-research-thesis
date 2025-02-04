#!/usr/bin/env python3.11

import pyaudio
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import gen_audio_ops as audio_ops
from datetime import datetime

# Definizione delle classi (gli indici corrispondono a:
# 0: background noise, 1: up, 2: down, 3: left, 4: right)
words = [
    '_background_noise_',
    'up',
    'down',
    'left',
    'right',
]

# Carica il modello addestrato (modifica il percorso se necessario)
model = keras.models.load_model("fully_trained.keras")

FORMAT = pyaudio.paFloat32
CHANNELS = 1
DESIRED_RATE = 16000  # Il sample rate che il modello si aspetta

audio = pyaudio.PyAudio()

# --- Selezione del dispositivo di input di default ---
default_device_info = audio.get_default_input_device_info()
default_device_index = default_device_info['index']
print("Utilizzo dispositivo di input di default:")
print(f"  Indice: {default_device_index}")
print(f"  Nome: {default_device_info['name']}")
RATE = int(default_device_info["defaultSampleRate"])
print(f"Sample rate del dispositivo: {RATE} Hz")

if RATE < DESIRED_RATE:
    print(f"Il sample rate del dispositivo è inferiore a {DESIRED_RATE} Hz. Verrà effettuato l'upsampling.")
elif RATE > DESIRED_RATE:
    print(f"Il sample rate del dispositivo è superiore a {DESIRED_RATE} Hz. Verrà effettuato il downsampling.")
else:
    print(f"Il sample rate del dispositivo è esattamente {DESIRED_RATE} Hz.")

# Impostazione del frames per buffer
FRAMES_PER_BUFFER = 8000  # Puoi modificare questo valore se necessario

# Buffer per contenere i campioni audio (inizialmente, 1 secondo di audio al RATE del dispositivo)
samples = np.zeros((RATE,))

def callback(input_data, frame_count, time_info, flags):
    global samples
    # Converte i dati in un array NumPy di float32
    new_samples = np.frombuffer(input_data, np.float32)
    samples = np.concatenate((samples, new_samples))
    samples = samples[-RATE:]
    
    if len(samples) == RATE:
        # Risample (upsample o downsample) per ottenere DESIRED_RATE campioni
        original_indices = np.arange(RATE)
        new_indices = np.linspace(0, RATE - 1, DESIRED_RATE)
        resampled_samples = np.interp(new_indices, original_indices, samples)
        
        # Normalizzazione: sottrai la media e scala in base al massimo assoluto
        normalised = resampled_samples - np.mean(resampled_samples)
        max_val = np.max(np.abs(normalised))
        if max_val > 0:
            normalised = normalised / max_val

        # Creazione dello spettrogramma:
        # Riformatta il segnale in un tensore di forma (DESIRED_RATE, 1)
        spectrogram = audio_ops.audio_spectrogram(
            np.reshape(normalised, (DESIRED_RATE, 1)),
            window_size=320,
            stride=160,
            magnitude_squared=True)
        
        # Riduci il numero di bin in frequenza tramite pooling
        spectrogram = tf.nn.pool(
            input=tf.expand_dims(spectrogram, -1),
            window_shape=[1, 6],
            strides=[1, 6],
            pooling_type='AVG',
            padding='SAME')
        spectrogram = tf.squeeze(spectrogram, axis=0)
        spectrogram = np.log10(spectrogram + 1e-6)
        
        # Adatta la forma dello spettrogramma all'input del modello.
        # FIXME: 
        input_tensor = np.reshape(spectrogram, (1, 99, 43, 1))
        
        # Effettua la predizione
        prediction = model.predict(input_tensor)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Costruisci il messaggio di stato da stampare sulla stessa riga.
        # Se viene rilevata una parola (predicted_class diverso da 0) e la confidenza è alta,
        # includi il nome della parola; altrimenti, mostra solo "Ascolto in corso..."
        if predicted_class != 0 and confidence > 0.9:
            status = f"Ascolto in corso... [rilevata parola {words[predicted_class]}]"
        else:
            status = "Ascolto in corso..."
        
        # Stampa il messaggio aggiornando la stessa riga
        print("\r" + status + " " * 20, end="", flush=True)
        
    return input_data, pyaudio.paContinue

stream = audio.open(
    input_device_index=default_device_index,
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    stream_callback=callback,
    frames_per_buffer=FRAMES_PER_BUFFER)

stream.start_stream()

while stream.is_active():
    time.sleep(0.1)
