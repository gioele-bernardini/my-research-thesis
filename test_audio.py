# sudo apt-get update
# sudo apt-get install portaudio19-dev python3-pyaudio
# pip install sounddevice matplotlib numpy

#!/usr/bin/env python3

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# Durata della registrazione in secondi
duration = 3  
# Frequenza di campionamento
fs = 44100  

print("Inizio registrazione...")
# Registra l'audio dal microfono
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
sd.wait()  # Attende la fine della registrazione
print("Registrazione completata.")

# Appiattisce l'array audio
audio = audio.flatten()

# Crea un array temporale per il grafico
time = np.linspace(0, duration, len(audio))

# Visualizza il grafico della forma d'onda
plt.plot(time, audio)
plt.xlabel('Tempo [s]')
plt.ylabel('Ampiezza')
plt.title('Forma d\'onda audio')
plt.show()
