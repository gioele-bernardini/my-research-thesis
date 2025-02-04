#!/usr/bin/env python3.11

import tensorflow as tf
import numpy as np
from tensorflow.io import gfile
import tensorflow_io as tfio
from tensorflow.python.ops import gen_audio_ops as audio_ops
from tqdm import tqdm
import matplotlib.pyplot as plt

# =======================================================================
#                          Configurazioni
# =======================================================================
SPEECH_DATA = 'speech_data'
EXPECTED_SAMPLES = 16000  # 1 secondo di audio a 16 kHz
NOISE_FLOOR = 0.05        # Soglia più bassa per trim dell'audio
MINIMUM_VOICE_LENGTH = int(0.1 * EXPECTED_SAMPLES)  # 0.1 secondi (~1600 campioni)

words = [
    '_background_noise_',
    'up',
    'down',
    'left',
    'right',
]

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1  # Non usato direttamente, si calcola dal resto

# =======================================================================
#                 Funzioni ausiliarie per gestione file
# =======================================================================
def get_files(word):
    """Ritorna la lista dei file .wav per una certa classe (word)."""
    return gfile.glob(f"{SPEECH_DATA}/{word}/*.wav")

# -----------------------------------------------------------------------
# Funzione per normalizzare la lunghezza dell'audio a EXACT 16000 campioni
# -----------------------------------------------------------------------
def fix_audio_length(audio, expected_length=EXPECTED_SAMPLES):
    """
    Se audio è più corto di expected_length, effettua padding a destra.
    Se è più lungo, estrae una finestra casuale di expected_length.
    """
    length = audio.shape[0]
    if length < expected_length:
        pad_amount = expected_length - length
        audio = tf.pad(audio, [[0, pad_amount]])
    elif length > expected_length:
        start = np.random.randint(0, length - expected_length)
        audio = audio[start:start+expected_length]
    return audio

# -----------------------------------------------------------------------
# Funzione per individuare inizio/fine voce usando trim
# -----------------------------------------------------------------------
def get_voice_position(audio, noise_floor=NOISE_FLOOR):
    """
    Ritorna (start, end) in campioni, cioè i punti in cui la voce
    inizia e finisce nell'audio normalizzato.
    """
    audio = audio - tf.reduce_mean(audio)
    max_val = tf.reduce_max(tf.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return tfio.audio.trim(audio, axis=0, epsilon=noise_floor)

# -----------------------------------------------------------------------
# Restituisce la durata della voce in campioni
# -----------------------------------------------------------------------
def get_voice_length(audio):
    """
    Ritorna la lunghezza (in campioni) della porzione di voce
    calcolata con get_voice_position.
    """
    voice_start, voice_end = get_voice_position(audio)
    return (voice_end - voice_start).numpy()

# -----------------------------------------------------------------------
# Check semplice per vedere se c'è abbastanza voce (~0.1s)
# -----------------------------------------------------------------------
def has_sufficient_voice(audio):
    return get_voice_length(audio) >= MINIMUM_VOICE_LENGTH

# =======================================================================
#              Funzione per creare lo spettrogramma (aggiornata)
# =======================================================================
def get_spectrogram(audio):
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    
    # Aggiungi una dimensione batch solo se l'audio è 1D
    if len(audio.shape) == 1:
        audio = tf.expand_dims(audio, axis=0)
    
    audio = audio - tf.reduce_mean(audio, axis=-1, keepdims=True)
    max_val = tf.reduce_max(tf.abs(audio), axis=-1, keepdims=True)
    audio = tf.where(max_val > 0, audio / max_val, audio)
    
    spectrogram = audio_ops.audio_spectrogram(
        audio,
        window_size=320,
        stride=160,
        magnitude_squared=True
    )
    
    spectrogram = tf.cast(spectrogram, tf.float32)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    
    spectrogram = tf.nn.pool(
        input=spectrogram,
        window_shape=[1, 6],
        strides=[1, 6],
        pooling_type='AVG',
        padding='SAME'
    )
    
    # Se la dimensione batch è 1, squeeze l'asse 0; altrimenti non lo fare
    if spectrogram.shape[0] == 1:
        spectrogram = tf.squeeze(spectrogram, axis=[0, 3])
    else:
        spectrogram = tf.squeeze(spectrogram, axis=3)
    
    spectrogram = tf.math.log(spectrogram + 1e-6) / tf.math.log(tf.constant(10, dtype=tf.float32))
    return spectrogram.numpy()

# =======================================================================
#   Processa un singolo file: carica audio, normalizza, shift + background
# =======================================================================
def process_file(file_path, label):
    """
    Carica l'audio, normalizza la lunghezza a 16000 campioni,
    centra la voce in un punto casuale limitato,
    e infine aggiunge rumore di background.
    Ritorna (spettrogramma, label).
    """
    # Caricamento
    audio_tensor = tfio.audio.AudioIOTensor(file_path)
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio = tf.squeeze(audio, axis=-1)  # Da [N,1] a [N]
    
    # Forza lunghezza a 16000 campioni
    audio = fix_audio_length(audio, EXPECTED_SAMPLES)
    
    # Se non è background e non c'è voce sufficiente, scarta
    if label != words.index('_background_noise_'):
        if not has_sufficient_voice(audio):
            return None
    
    # Individua la parte di voce
    voice_start, voice_end = get_voice_position(audio, NOISE_FLOOR)
    voice_start = int(voice_start.numpy())
    voice_end = int(voice_end.numpy())
    voice_length = voice_end - voice_start
    
    # Crea un buffer vuoto di 16000 campioni
    new_audio = np.zeros(EXPECTED_SAMPLES, dtype=np.float32)
    
    if voice_length > 0 and voice_length < EXPECTED_SAMPLES:
        max_offset = EXPECTED_SAMPLES - voice_length
        offset = np.random.randint(0, max_offset + 1)
        new_audio[offset:offset+voice_length] = audio[voice_start:voice_end]
    
    # Aggiunta di background noise leggero
    background_volume = np.random.uniform(0, 0.1)
    background_files = get_files('_background_noise_')
    if background_files:
        background_file = np.random.choice(background_files)
        background_tensor = tfio.audio.AudioIOTensor(background_file)
        background = tf.cast(background_tensor[:], tf.float32)
        background = tf.squeeze(background, axis=-1)
        
        if len(background) > EXPECTED_SAMPLES:
            start_bg = np.random.randint(0, len(background) - EXPECTED_SAMPLES)
            background = background[start_bg:start_bg+EXPECTED_SAMPLES]
        else:
            background = fix_audio_length(background, EXPECTED_SAMPLES)
        
        background = background - tf.reduce_mean(background)
        bmax = tf.reduce_max(tf.abs(background))
        if bmax > 0:
            background = background / bmax
        new_audio = new_audio + background_volume * background.numpy()
    
    spectrogram = get_spectrogram(new_audio)
    return (spectrogram, label)

# =======================================================================
# Processa una lista di file in batch
# =======================================================================
def process_files(file_names, label, repeat=1):
    """
    Per ogni file in 'file_names', genera 'repeat' augmentations
    e produce una lista di (spettrogramma, label).
    """
    samples = []
    repeated_list = tf.repeat(file_names, repeat).numpy()
    
    for fname in tqdm(repeated_list, desc=f"Processing label {label}", leave=False):
        result = process_file(fname, label)
        if result is not None:
            samples.append(result)
    return samples

# =======================================================================
#               Creazione liste train, validation, test
# =======================================================================
train = []
validate = []
test = []

# Conta file validi per bilanciare: per le parole vere, controlliamo quanti
# file hanno "abbastanza voce". Per _background_noise_ non si applica.
valid_counts = {}
for word in words:
    if word == '_background_noise_':
        continue
    all_files = get_files(word)
    valid_files = []
    for f in tqdm(all_files, desc=f"Checking '{word}'", leave=False):
        audio_tensor = tfio.audio.AudioIOTensor(f)
        audio = tf.cast(audio_tensor[:], tf.float32)
        audio = tf.squeeze(audio, axis=-1)
        audio = fix_audio_length(audio, EXPECTED_SAMPLES)
        if has_sufficient_voice(audio):
            valid_files.append(f)
    valid_counts[word] = len(valid_files)

max_count = max(valid_counts.values())
print("Valid counts per class:", valid_counts)
print("Max count among classes:", max_count)

# Processa i file di ogni parola (tranne _background_noise_) bilanciandoli
for word in words:
    if word == '_background_noise_':
        continue
    all_files = get_files(word)
    valid_files = []
    for f in tqdm(all_files, desc=f"Checking '{word}'", leave=False):
        audio_tensor = tfio.audio.AudioIOTensor(f)
        audio = tf.cast(audio_tensor[:], tf.float32)
        audio = tf.squeeze(audio, axis=-1)
        audio = fix_audio_length(audio, EXPECTED_SAMPLES)
        if has_sufficient_voice(audio):
            valid_files.append(f)
    
    repeat_factor = max(1, int(max_count / len(valid_files)))
    print(f"Processing '{word}' with repeat factor = {repeat_factor} (valid files: {len(valid_files)})")
    
    np.random.shuffle(valid_files)
    train_size = int(TRAIN_SIZE * len(valid_files))
    validation_size = int(VALIDATION_SIZE * len(valid_files))
    
    word_label = words.index(word)
    train.extend(process_files(valid_files[:train_size], word_label, repeat=repeat_factor))
    validate.extend(process_files(valid_files[train_size:train_size+validation_size], word_label, repeat=repeat_factor))
    test.extend(process_files(valid_files[train_size+validation_size:], word_label, repeat=repeat_factor))

print(f"After processing target words - Train: {len(train)}, Test: {len(test)}, Validate: {len(validate)}")

# =======================================================================
#  Gestione dei file di background noise (classe index = 0 se in words[0])
# =======================================================================
def process_background(file_name, label):
    """
    Spezza il file di background in chunk di 16000 campioni
    con stride di 16000 per evitare un eccesso di sample.
    Ritorna la lista di (spettrogramma, label).
    """
    audio_tensor = tfio.audio.AudioIOTensor(file_name)
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio = tf.squeeze(audio, axis=-1)
    audio_length = len(audio)
    
    samples = []
    stride = EXPECTED_SAMPLES  # 16000
    for section_start in range(0, audio_length - EXPECTED_SAMPLES, stride):
        section_end = section_start + EXPECTED_SAMPLES
        section = audio[section_start:section_end]
        section = fix_audio_length(section, EXPECTED_SAMPLES)
        spectrogram = get_spectrogram(section)
        samples.append((spectrogram, label))
    return samples

bg_label = words.index('_background_noise_')
bg_files = get_files('_background_noise_')

for file_name in tqdm(bg_files, desc="Processing Background Noise"):
    background_samples = process_background(file_name, bg_label)
    np.random.shuffle(background_samples)
    
    train_size = int(TRAIN_SIZE * len(background_samples))
    validation_size = int(VALIDATION_SIZE * len(background_samples))
    
    train.extend(background_samples[:train_size])
    validate.extend(background_samples[train_size:train_size+validation_size])
    test.extend(background_samples[train_size+validation_size:])

print(f"After processing background noise - Train: {len(train)}, Test: {len(test)}, Validate: {len(validate)}")

# =======================================================================
#     Shuffle finale e salvataggio su disco
# =======================================================================
np.random.shuffle(train)
X_train, Y_train = zip(*train)
X_validate, Y_validate = zip(*validate)
X_test, Y_test = zip(*test)

np.savez_compressed("training_spectrogram.npz", X=X_train, Y=Y_train)
print("Saved training data")
np.savez_compressed("validation_spectrogram.npz", X=X_validate, Y=Y_validate)
print("Saved validation data")
np.savez_compressed("test_spectrogram.npz", X=X_test, Y=Y_test)
print("Saved test data")

# Dimensioni immagine spettrale per la visualizzazione
IMG_WIDTH = X_train[0].shape[0]
IMG_HEIGHT = X_train[0].shape[1]

# =======================================================================
#   Funzione di debug/visualizzazione
# =======================================================================
def plot_images2(images_arr, imageWidth, imageHeight):
    fig, axes = plt.subplots(5, 5, figsize=(10, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(np.reshape(img, (imageWidth, imageHeight)), cmap='viridis')
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# Debug: visualizza qualche spettrogramma per ciascuna classe
for w in ['up', 'down', 'left', 'right']:
    w_idx = words.index(w)
    X_class = np.array(X_train)[np.array(Y_train) == w_idx]
    Y_class = np.array(Y_train)[np.array(Y_train) == w_idx]
    if len(X_class) > 20:
        plot_images2(X_class[:20], IMG_WIDTH, IMG_HEIGHT)
        print(f"{w} labels:", Y_class[:20])
    else:
        print(f"Fewer than 20 samples for {w} - skipping plot.")
