#!/usr/bin/env python3.11

import argparse
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.python.ops import gen_audio_ops as audio_ops

# Classi usate in training:
WORDS = [
    '_background_noise_',  # indice 0
    'up',                  # indice 1
    'down',                # indice 2
    'left',                # indice 3
    'right'                # indice 4
]

EXPECTED_SAMPLES = 16000  # 1 secondo di audio a 16 kHz

def load_audio(file_path, expected_samples=EXPECTED_SAMPLES):
    """
    Carica un file WAV (mono) con tfio.
    Assicura che l'uscita sia un tensore 1D di lunghezza esattamente expected_samples,
    tramite pad o trim.
    """
    # Caricamento con tfio (audio_tensor ha shape [N, channels])
    audio_tensor = tfio.audio.AudioIOTensor(file_path)
    audio = audio_tensor.to_tensor()  # shape [N, channels] in generale

    # Se ci sono più canali, selezioniamo solo il primo
    if audio.shape.rank == 2 and audio.shape[1] > 1:
        audio = audio[:, 0]  # shape [N]
    elif audio.shape.rank == 2 and audio.shape[1] == 1:
        # Se è [N,1], la riduciamo a [N]
        audio = tf.squeeze(audio, axis=1)

    # Ora audio è rank=1 => shape [N]
    audio = tf.cast(audio, tf.float32)

    num_samples = tf.shape(audio)[0]

    # Se l'audio è più corto dei 16000 campioni, facciamo pad.
    def pad_audio():
        pad_len = expected_samples - num_samples
        padding = tf.zeros([pad_len], dtype=tf.float32)  # shape [pad_len]
        return tf.concat([audio, padding], axis=0)       # shape [expected_samples]

    # Se l'audio è più lungo, lo tronchiamo.
    def trim_audio():
        return audio[:expected_samples]  # shape [expected_samples]

    audio = tf.cond(num_samples < expected_samples, pad_audio, trim_audio)
    # Ora audio ha shape [16000]

    return audio

def normalize_audio(audio):
    """
    Normalizza il segnale audio sottraendo la media e dividendo per il valore assoluto massimo.
    Evita divisioni per zero in caso di silenzio totale.
    """
    audio_mean = tf.math.reduce_mean(audio)
    audio = audio - audio_mean
    max_val = tf.math.reduce_max(tf.math.abs(audio))
    audio = tf.where(tf.equal(max_val, 0.0), audio, audio / max_val)
    return audio

def get_spectrogram(audio):
    """
    Genera lo spettrogramma replicando la pipeline del training:
      1) audio_ops.audio_spectrogram con window_size=320, stride=160, magnitude_squared=True
      2) tf.nn.pool con window_shape=[1, 6], strides=[1, 6], pooling_type='AVG', padding='SAME'
      3) log10(spectrogram + 1e-6)
    
    Il segnale audio in ingresso deve essere 1D (shape [16000]).
    """
    # Aggiungo dimensione batch per audio_ops.audio_spectrogram => shape [1, 16000]
    audio_2d = tf.expand_dims(audio, axis=0)

    # Crea lo spettrogramma
    spectrogram = audio_ops.audio_spectrogram(
        audio_2d,
        window_size=320,
        stride=160,
        magnitude_squared=True
    )
    # Ora spectrogram ha shape [1, frames, freq_bins]

    # Aggiungo un asse per canale => [1, frames, freq_bins, 1]
    spectrogram = tf.expand_dims(spectrogram, axis=-1)

    # Pooling per ridurre i freq bins
    spectrogram = tf.nn.pool(
        input=spectrogram,
        window_shape=[1, 6],   # riduce i freq_bins con fattore ~6
        strides=[1, 6],
        pooling_type='AVG',
        padding='SAME'
    )
    # Rimane [1, frames, freq_bins_reduced, 1]

    # Applico log10
    spectrogram = tf.math.log(spectrogram + 1e-6) / tf.math.log(tf.constant(10, dtype=tf.float32))

    # Per il modello, in training di solito si usava [batch, frames, freq_bins, 1].
    # Se vogliamo restituire un singolo 'sample' senza batch, togliamo dimensione batch:
    spectrogram = tf.squeeze(spectrogram, axis=0)  # => shape [frames, freq_bins_reduced, 1]

    return spectrogram

def preprocess_wav(file_path):
    """
    Carica il WAV, normalizza, e genera lo spettrogramma con la stessa pipeline del training.
    Restituisce tensore shape [frames, freq_bins, 1].
    """
    audio = load_audio(file_path, EXPECTED_SAMPLES)
    audio = normalize_audio(audio)
    spectro = get_spectrogram(audio)
    return spectro

def run_inference_tflite(tflite_model_path, spectrogram_tensor):
    """
    Esegue l'inferenza con un modello TFLite quantizzato.
      - 'spectrogram_tensor' ha forma [frames, freq_bins, 1].
      - Restituisce: (indice_classe_predetta, vettore_output).
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Aggiunge dimensione batch => [1, frames, freq_bins, 1]
    input_tensor = tf.expand_dims(spectrogram_tensor, axis=0)

    # Verifichiamo se l'input del modello è float32 o int8
    input_dtype = input_details[0]['dtype']
    input_tensor = tf.cast(input_tensor, input_dtype)

    # Imposta l'input del modello
    interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())

    # Invoca l'inferenza
    interpreter.invoke()

    # Estraggo l'output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Se l'output ha shape [1, 5], prendi output_data[0] => [5]
    predictions = output_data[0]

    # Argmax per la classe predetta
    predicted_class = int(np.argmax(predictions))

    return predicted_class, predictions

def main():
    parser = argparse.ArgumentParser(
        description="Esegui inferenza su un singolo WAV con modello TFLite quantizzato."
    )
    parser.add_argument("--wav", type=str, required=True, help="Percorso del file WAV mono 16kHz.")
    parser.add_argument("--model", type=str, required=True, help="Percorso del file .tflite quantizzato.")
    args = parser.parse_args()

    # 1) Preprocessing del WAV => spettrogramma
    spectrogram = preprocess_wav(args.wav)

    # 2) Inferenza col modello TFLite
    pred_class, logits = run_inference_tflite(args.model, spectrogram)

    # 3) Stampa risultati
    print("Logits/probabilità grezze:", logits)
    print(f"Classe predetta: {pred_class} -> '{WORDS[pred_class]}'")

if __name__ == "__main__":
    main()
