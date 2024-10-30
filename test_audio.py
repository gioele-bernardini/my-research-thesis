#!/usr/bin/env python3

"""
Script to record audio from the microphone and display its waveform.

Dependencies:
- PortAudio development files:
  - Install via package manager (e.g., for Ubuntu/Debian):
    sudo apt-get update
    sudo apt-get install portaudio19-dev python3-pyaudio
  - Required for audio input/output functionality.

- Python packages:
  - sounddevice
  - matplotlib
  - numpy
  - Install via pip:
    pip install sounddevice matplotlib numpy

Usage:
- Ensure that the necessary dependencies are installed.
- Run the script to record audio and visualize the waveform.
"""

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# Duration of the recording in seconds
duration = 3
# Sampling frequency
fs = 44100

print("Starting recording...")
# Record audio from the microphone
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
sd.wait()  # Wait until the recording is finished
print("Recording completed.")

# Flatten the audio array
audio = audio.flatten()

# Create a time array for plotting
time = np.linspace(0, duration, len(audio))

# Plot the audio waveform
plt.plot(time, audio)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')
plt.show()

