#!/usr/bin/env python3.11

import tensorflow as tf
import numpy as np
from tensorflow.io import gfile
import tensorflow_io as tfio
from tensorflow.python.ops import gen_audio_ops as audio_ops
from tqdm import tqdm
import matplotlib.pyplot as plt

# =======================================================================
#                          Costanti Globali
# =======================================================================
SPEECH_DATA = 'speech_data'
EXPECTED_SAMPLES = 16000  # 1 secondo di audio a 16kHz
NOISE_FLOOR = 0.05        # Soglia per il trim (modificata da 0.1 a 0.05)
MINIMUM_VOICE_LENGTH = int(0.1 * EXPECTED_SAMPLES)  # 0.1 secondi (~1600 campioni)

words = [
    '_background_noise_',
    'up',
    'down',
    'left',
    'right',
]

# =======================================================================
#              Funzioni ausiliarie per gestione file e audio
# =======================================================================

def get_files(word):
    """Ritorna la lista dei file .wav per una certa classe (word)."""
    return gfile.glob(f"{SPEECH_DATA}/{word}/*.wav")

# ---------------------------
# Gestione della lunghezza
# ---------------------------
def fix_audio_length(audio, expected_length=EXPECTED_SAMPLES):
    """
    Se l'audio è più corto di expected_length, effettua il padding a destra.
    Se è più lungo, estrae una finestra casuale di expected_length.
    """
    length = audio.shape[0]
    if length < expected_length:
        pad_amount = expected_length - length
        audio = tf.pad(audio, [[0, pad_amount]])
    elif length > expected_length:
        start = np.random.randint(0, length - expected_length + 1)
        audio = audio[start:start+expected_length]
    return audio

# ---------------------------
# Soglie e criteri di trim
# ---------------------------
def get_voice_position(audio, noise_floor=NOISE_FLOOR):
    """
    Ritorna (start, end) in campioni, cioè i punti in cui la voce
    inizia e finisce nell'audio normalizzato.
    Utilizza tf.reduce_mean e tf.reduce_max per la normalizzazione.
    """
    audio = audio - tf.reduce_mean(audio)
    max_val = tf.reduce_max(tf.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return tfio.audio.trim(audio, axis=0, epsilon=noise_floor)

def get_voice_length(audio, noise_floor=NOISE_FLOOR):
    position = get_voice_position(audio, noise_floor)
    return (position[1] - position[0]).numpy()

def is_voice_present(audio, noise_floor=NOISE_FLOOR, required_length=MINIMUM_VOICE_LENGTH):
    voice_length = get_voice_length(audio, noise_floor)
    return voice_length >= required_length

# ---------------------------
# Funzione per validare un file
# ---------------------------
def is_valid_file(file_name):
    audio_tensor = tfio.audio.AudioIOTensor(file_name)
    audio = tf.cast(audio_tensor[:], tf.float32)
    
    # Invece di scartare file di lunghezza errata, li correggiamo:
    audio = fix_audio_length(audio, EXPECTED_SAMPLES)
    
    # Normalizzazione per il controllo
    audio = audio - tf.reduce_mean(audio)
    max_val = tf.reduce_max(tf.abs(audio))
    if max_val > 0:
        audio = audio / max_val
        
    if not is_voice_present(audio, NOISE_FLOOR, MINIMUM_VOICE_LENGTH):
        return False
    
    return True

# =======================================================================
#              Funzione per creare lo spettrogramma
# (rimane invariata rispetto al codice 1)
# =======================================================================
def get_spectrogram(audio):
    # Normalizza l'audio
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    
    # Crea lo spettrogramma
    spectrogram = audio_ops.audio_spectrogram(
        audio,
        window_size=320,
        stride=160,
        magnitude_squared=True
    ).numpy()
    
    # Riduci il numero di bande di frequenza tramite pooling
    spectrogram = tf.nn.pool(
        input=tf.expand_dims(spectrogram, -1),
        window_shape=[1, 6],
        strides=[1, 6],
        pooling_type='AVG',
        padding='SAME'
    )
    spectrogram = tf.squeeze(spectrogram, axis=0)
    spectrogram = np.log10(spectrogram + 1e-6)
    
    return spectrogram

# =======================================================================
#       Processa un file audio e restituisce (spettrogramma, label)
# =======================================================================
def process_file(file_path):
    audio_tensor = tfio.audio.AudioIOTensor(file_path)
    audio = tf.cast(audio_tensor[:], tf.float32)
    
    # Se necessario, rimuoviamo dimensioni inutili
    if len(audio.shape) > 1:
        audio = tf.squeeze(audio, axis=-1)
    
    # Gestione della lunghezza: forziamo a 16000 campioni
    audio = fix_audio_length(audio, EXPECTED_SAMPLES)
    
    # Se non è background noise e non c'è sufficiente voce, scarta il file
    current_label = file_path.split('/')[-2]
    if words.index(current_label) != words.index('_background_noise_'):
        if not is_voice_present(audio, NOISE_FLOOR, MINIMUM_VOICE_LENGTH):
            return None
    
    # -------------------------------------------------------------
    # Metodo di posizionamento: copia la parte di voce in un buffer
    # -------------------------------------------------------------
    voice_pos = get_voice_position(audio, NOISE_FLOOR)
    voice_start = int(voice_pos[0].numpy())
    voice_end = int(voice_pos[1].numpy())
    voice_length = voice_end - voice_start
    
    # Crea un buffer vuoto di 16000 campioni
    new_audio = np.zeros(EXPECTED_SAMPLES, dtype=np.float32)
    if voice_length > 0 and voice_length < EXPECTED_SAMPLES:
        max_offset = EXPECTED_SAMPLES - voice_length
        offset = np.random.randint(0, max_offset + 1)
        new_audio[offset:offset+voice_length] = audio[voice_start:voice_end].numpy()
    else:
        # Se voice_length è 0 o non compatibile, usa l'audio già "fixato"
        new_audio = audio.numpy()
    
    # Aggiungi del background noise
    background_volume = np.random.uniform(0, 0.1)
    background_files = get_files('_background_noise_')
    if background_files:
        background_file = np.random.choice(background_files)
        background_tensor = tfio.audio.AudioIOTensor(background_file)
        if len(background_tensor) > EXPECTED_SAMPLES:
            background_start = np.random.randint(0, len(background_tensor) - EXPECTED_SAMPLES + 1)
            background = tf.cast(background_tensor[background_start:background_start+EXPECTED_SAMPLES], tf.float32)
        else:
            background = fix_audio_length(tf.cast(background_tensor[:], tf.float32), EXPECTED_SAMPLES)
        background = background - tf.reduce_mean(background)
        bmax = tf.reduce_max(tf.abs(background))
        if bmax > 0:
            background = background / bmax
        new_audio = new_audio + background_volume * background.numpy()
    
    # Restituisce lo spettrogramma
    return get_spectrogram(new_audio)

# =======================================================================
#    Processa una lista di file in spettrogrammi con il relativo label
# =======================================================================
def process_files(file_names, label, repeat=1):
    file_names = tf.repeat(file_names, repeat).numpy()
    results = []
    for fname in tqdm(file_names, desc=f"Processing label {label}", leave=False):
        result = process_file(fname)
        if result is not None:
            results.append((result, label))
    return results

# =======================================================================
#      Inizializzazione liste per train, validation e test
# =======================================================================
train = []
validate = []
test = []

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1  # Non usato direttamente, calcolato dal resto

# Bilanciamento dinamico basato sul conteggio dei file validi per classe (escluso background)
valid_counts = {}
for word in words:
    if word == '_background_noise_':
        continue
    valid_files = [f for f in tqdm(get_files(word), desc=f"Checking '{word}'", leave=False) if is_valid_file(f)]
    valid_counts[word] = len(valid_files)

max_count = max(valid_counts.values())
print("Valid counts per class:", valid_counts)
print("Max count among classes:", max_count)

# Processa ciascuna parola (escluso background) con fattore di ripetizione dinamico
for word in words:
    if word == '_background_noise_':
        continue
    valid_files = [f for f in tqdm(get_files(word), desc=f"Checking '{word}'", leave=False) if is_valid_file(f)]
    repeat = max(1, int(max_count / len(valid_files)))
    print(f"Processing '{word}' with repeat factor = {repeat} (valid files: {len(valid_files)}, max_count: {max_count})")
    
    np.random.shuffle(valid_files)
    train_size = int(TRAIN_SIZE * len(valid_files))
    validation_size = int(VALIDATION_SIZE * len(valid_files))
    
    train.extend(process_files(valid_files[:train_size], words.index(word), repeat=repeat))
    validate.extend(process_files(valid_files[train_size:train_size + validation_size], words.index(word), repeat=repeat))
    test.extend(process_files(valid_files[train_size + validation_size:], words.index(word), repeat=repeat))

print(f"After processing target words - Train: {len(train)}, Test: {len(test)}, Validate: {len(validate)}")

# =======================================================================
#            Processa i file di background noise
# =======================================================================
def process_background(file_name, label):
    audio_tensor = tfio.audio.AudioIOTensor(file_name)
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio = fix_audio_length(audio, EXPECTED_SAMPLES)
    samples = []
    
    for section_start in tqdm(range(0, len(audio) - EXPECTED_SAMPLES, 8000), desc=file_name, leave=False):
        section = audio[section_start:section_start+EXPECTED_SAMPLES]
        spectrogram = get_spectrogram(section.numpy())
        samples.append((spectrogram, label))
    
    np.random.shuffle(samples)
    
    train_size = int(TRAIN_SIZE * len(samples))
    validation_size = int(VALIDATION_SIZE * len(samples))
    
    train.extend(samples[:train_size])
    validate.extend(samples[train_size:train_size + validation_size])
    test.extend(samples[train_size + validation_size:])

for file_name in tqdm(get_files('_background_noise_'), desc="Processing Background Noise"):
    process_background(file_name, words.index("_background_noise_"))

print(f"After processing background noise - Train: {len(train)}, Test: {len(test)}, Validate: {len(validate)}")

# =======================================================================
#   Shuffle finale e separazione features/labels per il salvataggio
# =======================================================================
np.random.shuffle(train)
X_train, Y_train = zip(*train)
X_validate, Y_validate = zip(*validate)
X_test, Y_test = zip(*test)

# Salvataggio dei dati
np.savez_compressed("training_spectrogram.npz", X=X_train, Y=Y_train)
print("Saved training data")
np.savez_compressed("validation_spectrogram.npz", X=X_validate, Y=Y_validate)
print("Saved validation data")
np.savez_compressed("test_spectrogram.npz", X=X_test, Y=Y_test)
print("Saved test data")

# =======================================================================
#      Dimensioni dell'immagine spettrale per visualizzazione
# =======================================================================
IMG_WIDTH = X_train[0].shape[0]
print(IMG_WIDTH)
IMG_HEIGHT = X_train[0].shape[1]
print(IMG_HEIGHT)

# =======================================================================
#      Funzione per visualizzare una griglia di immagini spettrali
# =======================================================================
def plot_images2(images_arr, imageWidth, imageHeight):
    fig, axes = plt.subplots(5, 5, figsize=(10, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(np.reshape(img, (imageWidth, imageHeight)), cmap='viridis')
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# Visualizzazione per le classi: "up", "down", "left", "right"
for word in ['up', 'down', 'left', 'right']:
    word_index = words.index(word)
    X_class = np.array(X_train)[np.array(Y_train) == word_index]
    Y_class = np.array(Y_train)[np.array(Y_train) == word_index]
    plot_images2(X_class[:20], IMG_WIDTH, IMG_HEIGHT)
    print(f"{word} labels:", Y_class[:20])
