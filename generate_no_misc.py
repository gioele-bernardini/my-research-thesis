#!/usr/bin/env python3.11

import tensorflow as tf
import numpy as np
from tensorflow.io import gfile
import tensorflow_io as tfio
from tensorflow.python.ops import gen_audio_ops as audio_ops
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
SPEECH_DATA = 'speech_data'
EXPECTED_SAMPLES = 16000  # 1 second of audio at 16kHz
# Con 0.1 funziona
NOISE_FLOOR = 0.05
# Con 10 funziona
MINIMUM_VOICE_LENGTH = EXPECTED_SAMPLES / 10  # Minimum voice length required
words = [
    '_background_noise_',
    'up',
    'down',
    'left',
    'right',
]

# Funzione per ottenere tutti i file di una determinata "word"
def get_files(word):
    return gfile.glob(f"{SPEECH_DATA}/{word}/*.wav")

# ---------------------- FUNZIONI PER IL CONTROLLO DI VALIDITÀ ----------------------
# (se non ti servono, puoi rimuoverle)

# Ottiene la posizione (inizio/fine) della "voce" nell'audio, basandosi su un noise floor
def get_voice_position(audio, noise_floor):
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    return tfio.audio.trim(audio, axis=0, epsilon=noise_floor)

# Calcola la durata (in campioni) della parte "voce" nell'audio
def get_voice_length(audio, noise_floor):
    position = get_voice_position(audio, noise_floor)
    return (position[1] - position[0]).numpy()

# Controlla se nell'audio c'è una quantità sufficiente di "voce"
def is_voice_present(audio, noise_floor, required_length):
    voice_length = get_voice_length(audio, noise_floor)
    return voice_length >= required_length

# Verifica che l'audio abbia EXACT 16000 campioni (1 secondo)
def is_correct_length(audio, expected_length):
    return (audio.shape[0] == expected_length).numpy()

# Verifica che il file sia "valido" in base alla lunghezza e al contenuto di voce
def is_valid_file(file_name):
    audio_tensor = tfio.audio.AudioIOTensor(file_name)
    
    if not is_correct_length(audio_tensor, EXPECTED_SAMPLES):
        return False
    
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    
    if not is_voice_present(audio, NOISE_FLOOR, MINIMUM_VOICE_LENGTH):
        return False
    
    return True

# ---------------------- GENERAZIONE DELLO SPETTROGRAMMA ----------------------

def get_spectrogram(audio):
    """
    Prende un vettore audio 1D (già normalizzato) e ne calcola lo spettrogramma.
    Restituisce un array numpy 2D (frequenze x frame), 
    poi "poolato" per ridurre le dimensioni, 
    e infine trasformato in scala logaritmica.
    """
    # (Ri)normalizza l'audio (nel caso servisse, ma di solito è già normalizzato)
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    
    # Calcola lo spettrogramma
    spectrogram = audio_ops.audio_spectrogram(
        audio,
        window_size=320,
        stride=160,
        magnitude_squared=True
    ).numpy()
    
    # Riduce il numero di frequenze tramite pooling
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

# ---------------------- PROCESSAMENTO FILE (TARGET WORDS) ----------------------

def process_file(file_path):
    """
    Legge l'audio, lo normalizza e genera direttamente lo spettrogramma
    (senza aggiungere rumore né spostare l'audio).
    """
    audio_tensor = tfio.audio.AudioIOTensor(file_path)
    audio = tf.cast(audio_tensor[:], tf.float32)
    
    # Normalizzazione base
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    
    # Genera lo spettrogramma
    return get_spectrogram(audio)

def process_files(file_names, label, repeat=1):
    """
    process_file() su ogni file, con eventuale oversampling (repeat).
    Ritorna una lista di (spectrogram, label).
    """
    file_names = tf.repeat(file_names, repeat).numpy()
    results = []
    for file_name in tqdm(file_names, desc=f"Processing label={label}", leave=False):
        spectro = process_file(file_name)
        results.append((spectro, label))
    return results

# ---------------------- INIZIALIZZAZIONE LISTE ----------------------
train = []
validate = []
test = []

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1  # Il resto

# ---------------------- BILANCIAMENTO DINAMICO CLASSES (ESCL. BG) ----------------------
valid_counts = {}
for word in words:
    if word == '_background_noise_':
        continue
    # Filtra i file validi in base alle funzioni di check
    all_files = get_files(word)
    valid_files = [
        file_name
        for file_name in tqdm(all_files, desc=f"Checking '{word}'", leave=False)
        if is_valid_file(file_name)
    ]
    valid_counts[word] = len(valid_files)

max_count = max(valid_counts.values())
print("Valid counts per class:", valid_counts)
print("Max count among classes:", max_count)

# ---------------------- PROCESSAMENTO WORDS (ESCL. BG) ----------------------
for word in words:
    if word == '_background_noise_':
        continue
    
    valid_files = [
        file_name
        for file_name in tqdm(get_files(word), desc=f"Checking '{word}'", leave=False)
        if is_valid_file(file_name)
    ]
    # Ripetizione per bilanciare le classi
    repeat = max(1, int(max_count / len(valid_files)))
    
    print(f"Processing '{word}' with repeat factor = {repeat} (valid files: {len(valid_files)}, max_count: {max_count})")
    np.random.shuffle(valid_files)
    
    train_size = int(TRAIN_SIZE * len(valid_files))
    validation_size = int(VALIDATION_SIZE * len(valid_files))
    
    # Suddivisione e process_files
    train.extend(process_files(valid_files[:train_size], words.index(word), repeat=repeat))
    validate.extend(process_files(valid_files[train_size:train_size + validation_size], words.index(word), repeat=repeat))
    test.extend(process_files(valid_files[train_size + validation_size:], words.index(word), repeat=repeat))

print(f"After processing target words - Train: {len(train)}, Test: {len(test)}, Validate: {len(validate)}")

# ---------------------- PROCESSAMENTO BACKGROUND NOISE ----------------------
def process_background(file_name, label):
    """
    Divide il file di background in segmenti da 1 secondo (16000 campioni),
    crea per ognuno il relativo spettrogramma e lo aggiunge alle liste
    (train, validate, test) in base alle percentuali definite.
    """
    audio_tensor = tfio.audio.AudioIOTensor(file_name)
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio_length = len(audio)
    
    samples = []
    
    # Scorriamo l'audio a step di 8000 campioni (mezzo secondo), 
    # prendendo blocchi da 16000 campioni (1 secondo)
    for section_start in tqdm(range(0, audio_length - EXPECTED_SAMPLES, 8000),
                              desc=file_name, leave=False):
        section_end = section_start + EXPECTED_SAMPLES
        section = audio[section_start:section_end]
        
        # Normalizzazione e spettrogramma
        section = section - tf.reduce_mean(section)
        section = section / tf.reduce_max(tf.abs(section))
        spectrogram = get_spectrogram(section)
        samples.append((spectrogram, label))
    
    # Shuffle locale
    np.random.shuffle(samples)
    
    # Suddivisione train, validate, test
    t_size = int(TRAIN_SIZE * len(samples))
    v_size = int(VALIDATION_SIZE * len(samples))
    
    train.extend(samples[:t_size])
    validate.extend(samples[t_size:t_size + v_size])
    test.extend(samples[t_size + v_size:])

# Processa i file di background come un'unica "classe"
bg_files = get_files('_background_noise_')
for file_name in tqdm(bg_files, desc="Processing Background Noise"):
    process_background(file_name, words.index("_background_noise_"))

print(f"After processing background noise - Train: {len(train)}, Test: {len(test)}, Validate: {len(validate)}")

# ---------------------- RANDOMIZZA E SEPARA FEATURES/LABELS ----------------------
np.random.shuffle(train)
X_train, Y_train = zip(*train)
X_validate, Y_validate = zip(*validate)
X_test, Y_test = zip(*test)

# ---------------------- SALVATAGGIO ----------------------
np.savez_compressed("training_spectrogram.npz", X=X_train, Y=Y_train)
print("Saved training data")
np.savez_compressed("validation_spectrogram.npz", X=X_validate, Y=Y_validate)
print("Saved validation data")
np.savez_compressed("test_spectrogram.npz", X=X_test, Y=Y_test)
print("Saved test data")

# Otteniamo larghezza e altezza dello spettrogramma (in termini di "righe" e "colonne")
IMG_WIDTH = X_train[0].shape[0]
print("IMG_WIDTH:", IMG_WIDTH)
IMG_HEIGHT = X_train[0].shape[1]
print("IMG_HEIGHT:", IMG_HEIGHT)

# ---------------------- FUNZIONE DI VISUALIZZAZIONE ----------------------
def plot_images2(images_arr, imageWidth, imageHeight):
    fig, axes = plt.subplots(5, 5, figsize=(10, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(np.reshape(img, (imageWidth, imageHeight)), cmap='viridis')
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# ---------------------- ESEMPIO DI VISUALIZZAZIONE ----------------------
for word in ['up', 'down', 'left', 'right']:
    word_index = words.index(word)
    X_class = np.array(X_train)[np.array(Y_train) == word_index]
    Y_class = np.array(Y_train)[np.array(Y_train) == word_index]
    plot_images2(X_class[:20], IMG_WIDTH, IMG_HEIGHT)
    print(f"{word} labels:", Y_class[:20])
