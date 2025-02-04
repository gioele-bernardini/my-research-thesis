#!/usr/bin/env python3.11
"""
Questo script converte il modello Keras addestrato in un modello TFLite interamente quantizzato in INT8,
utilizzando una concrete function per fissare il grafo computazionale e definire correttamente le forme.
Dopo la conversione, esegue un test sul test set per verificare l'accuratezza del modello quantizzato.
Infine, genera un file C contenente il modello per il deployment embedded.
"""

import tensorflow as tf
import numpy as np
import os

# ================================
# 1. Caricamento e preparazione dati
# ================================
# Carica i dati degli spettrogrammi
training_spectrogram = np.load('training_spectrogram.npz', allow_pickle=True)
validation_spectrogram = np.load('validation_spectrogram.npz', allow_pickle=True)
test_spectrogram = np.load('test_spectrogram.npz', allow_pickle=True)

X_train = training_spectrogram['X']
X_validate = validation_spectrogram['X']
X_test = test_spectrogram['X']
Y_test = test_spectrogram['Y']  # etichette per il test

# Funzione per assicurarsi che ogni spettrogramma abbia la dimensione del canale (da (W, H) a (W, H, 1))
def ensure_channel_dimension(X):
    if len(X[0].shape) == 2:
        return np.array([np.expand_dims(x, axis=-1) for x in X])
    return np.array(X)

X_train = ensure_channel_dimension(X_train)
X_validate = ensure_channel_dimension(X_validate)
X_test = ensure_channel_dimension(X_test)

# Combina i dati (train+validate+test) per il dataset rappresentativo
complete_train_X = np.concatenate((X_train, X_validate, X_test))

# ================================
# 2. Caricamento e conversione del modello
# ================================
# Carica il modello addestrato (salvato con estensione .keras)
model = tf.keras.models.load_model("fully_trained.keras")
print("Modello Keras caricato correttamente.")

# Creazione della Concrete Function
# Se il batch size non è definito (None), lo imposto a 1 per la conversione.
input_shape = model.inputs[0].shape
if input_shape[0] is None:
    input_shape = (1,) + tuple(input_shape[1:])

concrete_func = tf.function(model).get_concrete_function(
    tf.TensorSpec(input_shape, model.inputs[0].dtype)
)
print("Concrete function creata con input shape:", input_shape)

# Configurazione del convertitore TFLite
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Definisci il generatore del dataset rappresentativo.
def representative_dataset_gen():
    # Scorriamo il dataset a step di 100, prelevando un campione per volta.
    for i in range(0, len(complete_train_X), 100):
        # complete_train_X[i:i+1] restituisce un array con shape (1, IMG_WIDTH, IMG_HEIGHT, 1)
        yield [complete_train_X[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset_gen

# Imposta il target per operazioni INT8: quantizza sia pesi che attivazioni
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Esegui la conversione
print("Avvio della conversione in TFLite INT8...")
tflite_int8_model = converter.convert()

# Salva il modello convertito su file
with open("converted_model_int8.tflite", "wb") as f:
    f.write(tflite_int8_model)
print("Modello TFLite INT8 salvato come 'converted_model_int8.tflite'.")

# (Opzionale) Genera un file C contenente il modello, utile per il deployment embedded.
os.system("xxd -i converted_model_int8.tflite > model_data.cc")
print("File C generato: 'model_data.cc'")

# ================================
# 3. Test del modello TFLite quantizzato
# ================================
print("Avvio del test sul test set per il modello TFLite INT8...")

# Inizializza l'interprete TFLite
interpreter = tf.lite.Interpreter(model_path="converted_model_int8.tflite")
interpreter.allocate_tensors()

# Ottieni i dettagli degli input e output
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Parametri di quantizzazione per l'input e l'output
input_scale, input_zero_point = input_details['quantization']
output_scale, output_zero_point = output_details['quantization']
print("Dettagli quantizzazione input: scale = {}, zero_point = {}".format(input_scale, input_zero_point))
print("Dettagli quantizzazione output: scale = {}, zero_point = {}".format(output_scale, output_zero_point))

# Esegui l'inferenza su ogni campione del test set e calcola l'accuratezza
correct_predictions = 0
num_samples = len(X_test)

for i in range(num_samples):
    # Prepara il campione: aggiungi la dimensione batch e converti in float32
    input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)
    # Quantizza l'input in base ai parametri ottenuti
    input_data_quant = input_data / input_scale + input_zero_point
    input_data_quant = np.clip(np.round(input_data_quant), -128, 127).astype(np.int8)
    
    interpreter.set_tensor(input_details['index'], input_data_quant)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    # L'argmax sui risultati quantizzati è affidabile poiché la scala lineare non ne altera l'ordine
    predicted_class = np.argmax(output_data[0])
    if predicted_class == Y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / num_samples
print("Accuratezza del modello TFLite INT8 sul test set: {:.2f}%".format(accuracy * 100))
