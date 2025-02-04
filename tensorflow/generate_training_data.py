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
NOISE_FLOOR = 0.1
MINIMUM_VOICE_LENGTH = EXPECTED_SAMPLES / 4  # Minimum voice length required
words = [
    '_background_noise_',
    'up',
    'down',
    'left',
    'right',
]

# Function to get all files for a given word
def get_files(word):
    return gfile.glob(f"{SPEECH_DATA}/{word}/*.wav")

# Function to get the voice position in the audio
def get_voice_position(audio, noise_floor):
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    return tfio.audio.trim(audio, axis=0, epsilon=noise_floor)

# Function to calculate the voice length in the audio
def get_voice_length(audio, noise_floor):
    position = get_voice_position(audio, noise_floor)
    return (position[1] - position[0]).numpy()

# Check if sufficient voice is present
def is_voice_present(audio, noise_floor, required_length):
    voice_length = get_voice_length(audio, noise_floor)
    return voice_length >= required_length

# Check if the audio is of the correct length
def is_correct_length(audio, expected_length):
    return (audio.shape[0] == expected_length).numpy()

# Check if the file is valid based on length and voice content
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

# Function to generate a spectrogram from the audio
def get_spectrogram(audio):
    # Normalize the audio
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    
    # Create the spectrogram
    spectrogram = audio_ops.audio_spectrogram(
        audio,
        window_size=320,
        stride=160,
        magnitude_squared=True
    ).numpy()
    
    # Reduce the number of frequency bins using pooling
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

# Process an audio file and return its spectrogram
def process_file(file_path):
    audio_tensor = tfio.audio.AudioIOTensor(file_path)
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    
    # Randomly reposition the audio within the sample
    voice_start, voice_end = get_voice_position(audio, NOISE_FLOOR)
    end_gap = len(audio) - voice_end
    random_offset = np.random.uniform(0, voice_start + end_gap)
    audio = np.roll(audio, -int(random_offset) + end_gap)
    
    # Add some random background noise
    background_volume = np.random.uniform(0, 0.1)
    background_files = get_files('_background_noise_')
    background_file = np.random.choice(background_files)
    background_tensor = tfio.audio.AudioIOTensor(background_file)
    background_start = np.random.randint(0, len(background_tensor) - 16000)
    
    background = tf.cast(background_tensor[background_start:background_start + 16000], tf.float32)
    background = background - np.mean(background)
    background = background / np.max(np.abs(background))
    
    # Mix the original audio with the background noise
    audio = audio + background_volume * background
    
    # Return the spectrogram
    return get_spectrogram(audio)

# Process a list of files into spectrograms with the given label
def process_files(file_names, label, repeat=1):
    file_names = tf.repeat(file_names, repeat).numpy()
    return [(process_file(file_name), label) for file_name in tqdm(file_names, desc=f"Processing label {label}", leave=False)]

# Initialize training, validation, and test lists
train = []
validate = []
test = []

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1  # Not used directly, as it's computed from the rest

# Dynamic balancing based on valid file counts per class (excluding background noise)
valid_counts = {}
for word in words:
    if word == '_background_noise_':
        continue
    valid_files = [file_name for file_name in tqdm(get_files(word), desc=f"Checking '{word}'", leave=False) if is_valid_file(file_name)]
    valid_counts[word] = len(valid_files)

max_count = max(valid_counts.values())
print("Valid counts per class:", valid_counts)
print("Max count among classes:", max_count)

# Process each word (excluding background noise) with dynamic repetition factor
for word in words:
    if word == '_background_noise_':
        continue
    valid_files = [file_name for file_name in tqdm(get_files(word), desc=f"Checking '{word}'", leave=False) if is_valid_file(file_name)]
    repeat = max(1, int(max_count / len(valid_files)))
    print(f"Processing '{word}' with repeat factor = {repeat} (valid files: {len(valid_files)}, max_count: {max_count})")
    
    np.random.shuffle(valid_files)
    train_size = int(TRAIN_SIZE * len(valid_files))
    validation_size = int(VALIDATION_SIZE * len(valid_files))
    
    train.extend(process_files(valid_files[:train_size], words.index(word), repeat=repeat))
    validate.extend(process_files(valid_files[train_size:train_size + validation_size], words.index(word), repeat=repeat))
    test.extend(process_files(valid_files[train_size + validation_size:], words.index(word), repeat=repeat))

print(f"After processing target words - Train: {len(train)}, Test: {len(test)}, Validate: {len(validate)}")

# Process background noise files
def process_background(file_name, label):
    audio_tensor = tfio.audio.AudioIOTensor(file_name)
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio_length = len(audio)
    samples = []
    
    for section_start in tqdm(range(0, audio_length - EXPECTED_SAMPLES, 8000), desc=file_name, leave=False):
        section_end = section_start + EXPECTED_SAMPLES
        section = audio[section_start:section_end]
        spectrogram = get_spectrogram(section)
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

# Randomize training samples and separate features and labels
np.random.shuffle(train)
X_train, Y_train = zip(*train)
X_validate, Y_validate = zip(*validate)
X_test, Y_test = zip(*test)

# Save computed data
np.savez_compressed("training_spectrogram.npz", X=X_train, Y=Y_train)
print("Saved training data")
np.savez_compressed("validation_spectrogram.npz", X=X_validate, Y=Y_validate)
print("Saved validation data")
np.savez_compressed("test_spectrogram.npz", X=X_test, Y=Y_test)
print("Saved test data")

# Get the width and height of the spectrogram "image"
IMG_WIDTH = X_train[0].shape[0]
print(IMG_WIDTH)
IMG_HEIGHT = X_train[0].shape[1]
print(IMG_HEIGHT)

# Function to plot a grid of spectrogram images
def plot_images2(images_arr, imageWidth, imageHeight):
    fig, axes = plt.subplots(5, 5, figsize=(10, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(np.reshape(img, (imageWidth, imageHeight)), cmap='viridis')
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# Visualization for each class: "up", "down", "left", "right"
for word in ['up', 'down', 'left', 'right']:
    word_index = words.index(word)
    X_class = np.array(X_train)[np.array(Y_train) == word_index]
    Y_class = np.array(Y_train)[np.array(Y_train) == word_index]
    plot_images2(X_class[:20], IMG_WIDTH, IMG_HEIGHT)
    print(f"{word} labels:", Y_class[:20])
