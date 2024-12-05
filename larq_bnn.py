#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
import larq as lq
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence

# Verifica che TensorFlow veda la GPU (se disponibile)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Carica la lista dei comandi da 'commands_list.txt'
with open('commands_list.txt', 'r') as f:
    commands = f.read().splitlines()

print("Comandi per il training:", commands)

# Imposta i parametri principali
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Percorso del dataset
DATASET_PATH = 'speech_commands'

# Funzione per caricare gli audio e le etichette
def load_audio_files(commands, dataset_path):
    audio_paths = []
    labels = []
    for label, command in enumerate(commands):
        command_path = os.path.join(dataset_path, command)
        if os.path.exists(command_path):
            for file in os.listdir(command_path):
                if file.endswith('.wav'):
                    audio_paths.append(os.path.join(command_path, file))
                    labels.append(label)
        else:
            print(f"Attenzione: il comando '{command}' non esiste nel dataset.")
    return audio_paths, labels

audio_paths, labels = load_audio_files(commands, DATASET_PATH)

print(f"Numero totale di campioni: {len(audio_paths)}")

# Divide i dati in training e validation set
from sklearn.model_selection import train_test_split

train_paths, val_paths, train_labels, val_labels = train_test_split(
    audio_paths, labels, test_size=0.2, random_state=seed, stratify=labels)

# Funzione per estrarre gli spettrogrammi dagli audio
def paths_to_spectrograms(paths):
    spectrograms = []
    for path in paths:
        audio_binary = tf.io.read_file(path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=-1)
        # Pad o trunca l'audio a 1 secondo (16000 campioni)
        audio = audio[:16000]
        zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrograms.append(spectrogram)
    return spectrograms

# Prepara gli spettrogrammi per training e validation set
train_spectrograms = paths_to_spectrograms(train_paths)
val_spectrograms = paths_to_spectrograms(val_paths)

# Converte le liste in tensori
train_spectrograms = tf.stack(train_spectrograms)
val_spectrograms = tf.stack(val_spectrograms)
train_labels = tf.convert_to_tensor(train_labels)
val_labels = tf.convert_to_tensor(val_labels)

# Definisci il modello binarizzato
def create_binary_model(input_shape, num_classes):
    kwargs = dict(input_quantizer="ste_sign",
                  kernel_quantizer="ste_sign",
                  kernel_constraint="weight_clip")

    model = keras.models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Aggiungi livelli binarizzati
    model.add(lq.layers.QuantConv2D(32, kernel_size=3, padding='same', **kwargs))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.BatchNormalization())
    model.add(lq.layers.QuantConv2D(64, kernel_size=3, padding='same', **kwargs))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(lq.layers.QuantDense(128, **kwargs))
    model.add(layers.BatchNormalization())
    model.add(lq.layers.QuantDense(num_classes, **kwargs))
    model.add(layers.Activation('softmax'))

    return model

# Ottieni la forma dell'input
input_shape = train_spectrograms.shape[1:]
num_classes = len(commands)

# Crea il modello
model = create_binary_model(input_shape, num_classes)

# Compila il modello
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Mostra il sommario del modello
model.summary()

# Allena il modello
history = model.fit(train_spectrograms, train_labels,
                    validation_data=(val_spectrograms, val_labels),
                    epochs=20,
                    batch_size=32)

# Salva il modello
model.save('binary_kws_model.h5')
