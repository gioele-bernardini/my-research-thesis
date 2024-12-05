import os
import numpy as np
import tensorflow as tf
import larq as lq
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Verifica che TensorFlow rilevi la GPU (se disponibile)
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
DATASET_PATH = 'speech-commands/'

def load_audio_files(commands, dataset_path):
    """
    Carica i percorsi dei file audio e le relative etichette per i comandi specificati.
    
    Args:
        commands (list): Lista dei comandi da includere nel training.
        dataset_path (str): Percorso alla directory del dataset.

    Returns:
        tuple: (lista dei percorsi audio, lista delle etichette)
    """
    audio_paths = []
    labels = []
    for label, command in enumerate(commands):
        command_path = os.path.join(dataset_path, command)
        if os.path.exists(command_path) and os.path.isdir(command_path):
            wav_files = [f for f in os.listdir(command_path) if f.endswith('.wav')]
            if len(wav_files) == 0:
                print(f"Attenzione: la directory per il comando '{command}' è vuota.")
            for file in wav_files:
                audio_paths.append(os.path.join(command_path, file))
                labels.append(label)
        else:
            print(f"Attenzione: il comando '{command}' non esiste nel dataset.")
    return audio_paths, labels

audio_paths, labels = load_audio_files(commands, DATASET_PATH)

print(f"Numero totale di campioni: {len(audio_paths)}")

if len(audio_paths) == 0:
    raise ValueError("Nessun campione trovato. Verifica che le directory dei comandi siano corrette e contengano file '.wav'.")

# Suddividi i dati in set di training e validation
train_paths, val_paths, train_labels, val_labels = train_test_split(
    audio_paths, labels, test_size=0.2, random_state=seed, stratify=labels)

def paths_to_spectrograms(paths):
    """
    Converte i percorsi dei file audio in spettrogrammi.
    
    Args:
        paths (list): Lista dei percorsi dei file audio.

    Returns:
        tf.Tensor: Tensore contenente gli spettrogrammi.
    """
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
        spectrogram = tf.expand_dims(spectrogram, -1)  # Aggiungi una dimensione per il canale
        spectrograms.append(spectrogram)
    return spectrograms

# Prepara gli spettrogrammi per training e validation set
print("Convertendo i percorsi dei file audio in spettrogrammi...")
train_spectrograms = paths_to_spectrograms(train_paths)
val_spectrograms = paths_to_spectrograms(val_paths)

# Converte le liste in tensori
train_spectrograms = tf.stack(train_spectrograms)
val_spectrograms = tf.stack(val_spectrograms)
train_labels = tf.convert_to_tensor(train_labels)
val_labels = tf.convert_to_tensor(val_labels)

print(f"Forma degli spettrogrammi di training: {train_spectrograms.shape}")
print(f"Forma degli spettrogrammi di validation: {val_spectrograms.shape}")

# Definisci il modello binarizzato
def create_binary_model(input_shape, num_classes):
    """
    Crea una rete neurale binaria utilizzando Larq.
    
    Args:
        input_shape (tuple): Forma dell'input (altezza, larghezza, canali).
        num_classes (int): Numero di classi di output.

    Returns:
        keras.Model: Modello compilato.
    """
    # Parametri di quantizzazione per Larq
    quant_params = dict(
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip"
    )

    model = keras.models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Primo livello convoluzionale binarizzato
    model.add(lq.layers.QuantConv2D(32, kernel_size=3, padding='same', **quant_params))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=2))

    # Secondo livello convoluzionale binarizzato
    model.add(lq.layers.QuantConv2D(64, kernel_size=3, padding='same', **quant_params))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=2))

    # Livello di flattening
    model.add(layers.Flatten())

    # Primo livello denso binarizzato
    model.add(lq.layers.QuantDense(128, **quant_params))
    model.add(layers.Activation('relu'))

    # Livello di output binarizzato
    model.add(lq.layers.QuantDense(num_classes, **quant_params))
    model.add(layers.Activation('softmax'))

    return model

# Ottieni la forma dell'input e il numero di classi
input_shape = train_spectrograms.shape[1:]  # (altezza, larghezza, canali)
num_classes = len(commands)

print(f"Forma dell'input: {input_shape}")
print(f"Numero di classi: {num_classes}")

# Crea il modello
model = create_binary_model(input_shape, num_classes)

# Compila il modello
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Mostra il sommario del modello
model.summary()

# Allena il modello
history = model.fit(
    train_spectrograms, train_labels,
    validation_data=(val_spectrograms, val_labels),
    epochs=20,
    batch_size=32
)

# Salva il modello
model.save('binary_kws_model.h5')
print("Modello salvato come 'binary_kws_model.h5'")
