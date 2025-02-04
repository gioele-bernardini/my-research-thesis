#!/usr/bin/env python3
"""
Script per addestrare una rete CNN per il riconoscimento vocale
su 4 parole: 'up', 'down', 'left', 'right' (con il rumore di fondo come quinta classe).
Questo script non è destinato ad essere eseguito in ambiente Jupyter.
"""

import os
import shutil
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.data import Dataset
import matplotlib.pyplot as plt

# --- Impostazioni e caricamento dei dati ---

# Lista delle classi: la prima è il rumore di fondo, le altre sono le 4 parole target.
words = [
    '_background_noise_',  # indice 0
    'up',                  # indice 1
    'down',                # indice 2
    'left',                # indice 3
    'right'                # indice 4
]

# Carica i dati degli spettrogrammi e le etichette dai file pre-processati.
training_spectrogram = np.load('training_spectrogram.npz', allow_pickle=True)
validation_spectrogram = np.load('validation_spectrogram.npz', allow_pickle=True)
test_spectrogram = np.load('test_spectrogram.npz', allow_pickle=True)

X_train = training_spectrogram['X']
Y_train = training_spectrogram['Y']
X_validate = validation_spectrogram['X']
Y_validate = validation_spectrogram['Y']
X_test = test_spectrogram['X']
Y_test = test_spectrogram['Y']

# Determina le dimensioni degli spettrogrammi.
IMG_WIDTH = X_train[0].shape[0]
IMG_HEIGHT = X_train[0].shape[1]
num_classes = len(words)

# Se manca il canale, aggiungilo (da (W, H) a (W, H, 1)).
def ensure_channel_dimension(X):
    if len(X[0].shape) == 2:
        return np.array([np.expand_dims(x, axis=-1) for x in X])
    return np.array(X)

X_train = ensure_channel_dimension(X_train)
X_validate = ensure_channel_dimension(X_validate)
X_test = ensure_channel_dimension(X_test)

# --- Analisi della distribuzione delle classi ---
plt.figure()
plt.hist(Y_train, bins=range(0, num_classes + 1), align='left')
plt.title("Distribuzione delle classi originali")
plt.xlabel("Classe")
plt.ylabel("Conteggio")
plt.savefig("distribution_original.png")  # Salva il grafico su file.
plt.close()

unique, counts = np.unique(Y_train, return_counts=True)
print("Distribuzione delle classi:", dict(zip([words[i] for i in unique], counts)))

# --- Creazione dei dataset tf.data ---
batch_size = 30

train_dataset = Dataset.from_tensor_slices((X_train, Y_train)).repeat().shuffle(len(X_train)).batch(batch_size)
validation_dataset = Dataset.from_tensor_slices((X_validate, Y_validate)).batch(X_validate.shape[0])
test_dataset = Dataset.from_tensor_slices((X_test, Y_test)).batch(len(X_test))

# --- Definizione del modello ---
model = Sequential([
    Conv2D(4, kernel_size=3,
           padding='same',
           activation='relu',
           kernel_regularizer=regularizers.l2(0.001),
           name='conv_layer1',
           input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    MaxPooling2D(name='max_pooling1', pool_size=(2, 2)),
    Conv2D(4, kernel_size=3,
           padding='same',
           activation='relu',
           kernel_regularizer=regularizers.l2(0.001),
           name='conv_layer2'),
    MaxPooling2D(name='max_pooling2', pool_size=(2, 2)),
    Flatten(),
    Dropout(0.2),
    Dense(40,
          activation='relu',
          kernel_regularizer=regularizers.l2(0.001),
          name='hidden_layer1'),
    Dense(num_classes,
          activation='softmax',
          kernel_regularizer=regularizers.l2(0.001),
          name='output')
])
model.summary()

# --- Compilazione del modello ---
epochs = 30
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# --- Gestione della directory dei log ---
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if os.path.exists("logs"):
    # (Opzionale) pulizia della cartella dei log precedenti.
    shutil.rmtree("logs")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Callback per salvare il miglior modello.
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoint.keras",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# --- Addestramento iniziale ---
history = model.fit(
    train_dataset,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=validation_dataset,
    validation_steps=1,
    callbacks=[tensorboard_callback, model_checkpoint_callback]
)

# Salva il modello addestrato.
model.save("trained.keras")

# --- Valutazione e predizione sul test set ---
# Carica il modello migliore salvato in base a val_accuracy.
model2 = keras.models.load_model("checkpoint.keras")
results = model2.evaluate(X_test, Y_test, batch_size=128)
print("Risultati sul test set:", results)

# Predizioni e matrice di confusione.
predictions = model2.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
cm = tf.math.confusion_matrix(Y_test, predicted_classes)
print("Matrice di confusione:\n", cm.numpy())

# TODO: fix Fine tuning
# --- Addestramento sul dataset completo (train+validate+test) ---
complete_train_X = np.concatenate((X_train, X_validate, X_test))
complete_train_Y = np.concatenate((Y_train, Y_validate, Y_test))
complete_train_dataset = Dataset.from_tensor_slices((complete_train_X, complete_train_Y)).repeat().shuffle(300000).batch(batch_size)

history = model2.fit(
    complete_train_dataset,
    steps_per_epoch=len(complete_train_X) // batch_size,
    epochs=5
)

# Valutazione finale sul dataset completo.
predictions = model2.predict(complete_train_X)
predicted_classes = np.argmax(predictions, axis=1)
cm = tf.math.confusion_matrix(complete_train_Y, predicted_classes)
print("Matrice di confusione (dataset completo):\n", cm.numpy())

# Salva il modello "finitamente" addestrato, utilizzando l'estensione .keras.
model2.save("fully_trained.keras")

# # --- (Opzionale) Conversione del modello in TFLite per MCU ---
# converter = tf.lite.TFLiteConverter.from_keras_model(model2)
# # Per microcontrollori, potrebbero essere utili ulteriori ottimizzazioni (es. quantizzazione).
# tflite_model = converter.convert()
# with open("model.tflite", "wb") as f:
#     f.write(tflite_model)
# print("Modello convertito in TFLite salvato come model.tflite")

# if __name__ == '__main__':
#     # Esecuzione diretta dello script.
#     pass
