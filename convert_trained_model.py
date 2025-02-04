#!/usr/bin/env python3.11

import tensorflow as tf
import numpy as np
import os

# Carica i dati degli spettrogrammi
training_spectrogram = np.load('training_spectrogram.npz', allow_pickle=True)
validation_spectrogram = np.load('validation_spectrogram.npz', allow_pickle=True)
test_spectrogram = np.load('test_spectrogram.npz', allow_pickle=True)

X_train = training_spectrogram['X']
X_validate = validation_spectrogram['X']
X_test = test_spectrogram['X']

# Assicura che ogni spettrogramma abbia la dimensione del canale (da (W, H) a (W, H, 1))
def ensure_channel_dimension(X):
    if len(X[0].shape) == 2:
        return np.array([np.expand_dims(x, axis=-1) for x in X])
    return np.array(X)

X_train = ensure_channel_dimension(X_train)
X_validate = ensure_channel_dimension(X_validate)
X_test = ensure_channel_dimension(X_test)

# Combina tutti i dati in un unico array, che verrà usato per il dataset rappresentativo
complete_train_X = np.concatenate((X_train, X_validate, X_test))

# Carica il modello addestrato (salvato con estensione .keras)
model = tf.keras.models.load_model("fully_trained.keras")

# Crea una concrete function a partire dal modello.
# Se il batch size non è definito (None), lo imposto a 1 per la conversione.
input_shape = model.inputs[0].shape
if input_shape[0] is None:
    input_shape = (1,) + tuple(input_shape[1:])

concrete_func = tf.function(model).get_concrete_function(
    tf.TensorSpec(input_shape, model.inputs[0].dtype)
)

# Crea il convertitore TFLite utilizzando la concrete function
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Definisci il generatore del dataset rappresentativo.
# Il generatore restituisce campioni con batch size 1, in modo da calibrare correttamente il range dei dati.
def representative_dataset_gen():
    # Scorriamo il dataset a step di 100, prelevando un solo campione per volta.
    for i in range(0, len(complete_train_X), 100):
        # complete_train_X[i:i+1] restituisce un array con shape (1, IMG_WIDTH, IMG_HEIGHT, 1)
        yield [complete_train_X[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset_gen

# Imposta il target per operazioni INT8: questo farà sì che sia i pesi che le attivazioni vengano quantizzati a INT8.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Imposta anche il tipo di input e output per l'inferenza (full integer quantization)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Esegui la conversione
tflite_int8_model = converter.convert()

# Salva il modello convertito su file
with open("converted_model_int8.tflite", "wb") as f:
    f.write(tflite_int8_model)

# (Opzionale) Genera un file C contenente il modello, utile per il deployment embedded.
os.system("xxd -i converted_model_int8.tflite > model_data.cc")

print("Conversione in INT8 completata: il modello TFLite è stato salvato come 'converted_model_int8.tflite'")
