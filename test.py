#!/usr/bin/env python3.11

import pyaudio
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import gen_audio_ops as audio_ops
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disabilita i log di TensorFlow

# Definisci l'array delle parole (l'indice 0 corrisponde al background noise)
words = [
    '_background_noise_',
    'up',
    'down',
    'left',
    'right',
]

# Carica il modello addestrato
model = keras.models.load_model("fully_trained.keras")

FORMAT = pyaudio.paFloat32
CHANNELS = 1
DESIRED_RATE = 16000  # Il sample rate richiesto dal modello

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
FRAMES_PER_BUFFER = 8000  # Puoi modificarlo se necessario

# Buffer per contenere i campioni audio (inizialmente 1 secondo di audio)
samples = np.zeros((RATE,), dtype=np.float32)

# Soglia di confidenza per la stampa della parola rilevata
CONFIDENCE_THRESHOLD = 0.7

# Cooldown in secondi tra un riconoscimento e l'altro (puoi impostarlo a 0.0 se non ti serve)
COOLDOWN_SECONDS = 1.0
last_detection_time = 0.0

# Funzione per silenziare temporaneamente la stdout (per nascondere i log di TensorFlow, se desideri)
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def callback(input_data, frame_count, time_info, flags):
    global samples
    global last_detection_time

    # Converte i dati grezzi in float32
    new_samples = np.frombuffer(input_data, dtype=np.float32)

    # Accumula e tieni solo gli ultimi 'RATE' campioni
    samples = np.concatenate((samples, new_samples))
    samples = samples[-RATE:]

    # Se non è ancora passato il cooldown, salta la classificazione
    if (time.time() - last_detection_time) < COOLDOWN_SECONDS:
        return (input_data, pyaudio.paContinue)

    # Se abbiamo esattamente RATE campioni, esegui la classificazione
    if len(samples) == RATE:
        # Resampling a DESIRED_RATE
        original_indices = np.arange(RATE)
        new_indices = np.linspace(0, RATE - 1, DESIRED_RATE)
        resampled_samples = np.interp(new_indices, original_indices, samples)

        # Normalizzazione
        normalised = resampled_samples - np.mean(resampled_samples)
        max_val = np.max(np.abs(normalised))
        if max_val > 0:
            normalised /= max_val

        # Calcolo spettrogramma
        spectrogram = audio_ops.audio_spectrogram(
            np.reshape(normalised, (DESIRED_RATE, 1)),
            window_size=320,
            stride=160,
            magnitude_squared=True
        )

        # Pooling
        spectrogram = tf.nn.pool(
            input=tf.expand_dims(spectrogram, -1),
            window_shape=[1, 6],
            strides=[1, 6],
            pooling_type='AVG',
            padding='SAME'
        )
        spectrogram = tf.squeeze(spectrogram, axis=0)
        spectrogram = np.log10(spectrogram + 1e-6)

        input_tensor = np.reshape(spectrogram, (1, 99, 43, 1))

        # Silenzia temporaneamente TensorFlow
        with HiddenPrints():
            prediction = model.predict(input_tensor)

        # prediction è un array di forma (1, len(words)); prendiamo la riga 0
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])

        # Se la parola non è background e la confidenza supera la soglia, stampiamo
        if predicted_class != 0 and confidence >= CONFIDENCE_THRESHOLD:
            print(f"\nPredicted Keyword: {words[predicted_class]} (Confidence: {confidence*100:.2f}%)")
            print(f"Logits: {prediction[0]}")
            
            # Reset del buffer per evitare multiple detection consecutive
            samples = np.zeros((RATE,), dtype=np.float32)
            last_detection_time = time.time()

    return (input_data, pyaudio.paContinue)

stream = audio.open(
    input_device_index=default_device_index,
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    stream_callback=callback,
    frames_per_buffer=FRAMES_PER_BUFFER
)

print("Listening... Press Ctrl+C to stop.")
stream.start_stream()

try:
    while stream.is_active():
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nInference stopped.")

stream.stop_stream()
stream.close()
audio.terminate()
