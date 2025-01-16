#!/usr/bin/env python3

import os
import numpy as np

def generate_headers(weights_dir='weights', headers_dir='headers'):
  if not os.path.exists(headers_dir):
    os.makedirs(headers_dir)

  for filename in os.listdir(weights_dir):
    file_path = os.path.join(weights_dir, filename)

    # Ignora eventuali file non .bin
    if not filename.endswith('.bin'):
      continue

    # Determina se è binarizzato o float in base al suffisso
    if '_binarized.bin' in filename:
      data_type = 'binarized'
    elif '_float.bin' in filename:
      data_type = 'float'
    else:
      # Non sappiamo che tipo sia, saltiamo o gestiamo come preferisci.
      print(f"Attenzione: {filename} non contiene né 'binarized' né 'float'.")
      continue

    # Genera un nome "base" senza estensione e suffissi vari
    base_name = os.path.splitext(filename)[0]  # rimuove .bin
    array_name = base_name.replace('.', '_')   # sostituisce eventuali punti

    # Crea il path per il file header
    header_filename = os.path.join(headers_dir, f'{base_name}.h')

    # Legge il file .bin in base al tipo di dati
    if data_type == 'binarized':
      with open(file_path, 'rb') as f:
        raw = f.read()
      # I dati sono bit-pack di 0/1 => array di uint8
      data = np.frombuffer(raw, dtype=np.uint8)
    else:
      # data_type == 'float'
      data = np.fromfile(file_path, dtype=np.float32)

    # Costruiamo il contenuto dell'header
    lines = []
    lines.append(f'// Header generato automaticamente per {filename}')
    lines.append(f'#ifndef {array_name.upper()}_H')
    lines.append(f'#define {array_name.upper()}_H\n')

    if data_type == 'binarized':
      # Array di unsigned char (byte)
      lines.append(f'static const unsigned char {array_name}[] = {{')
      hex_values = [f'0x{val:02X}' for val in data]
      for i in range(0, len(hex_values), 12):
        chunk = ', '.join(hex_values[i:i+12])
        lines.append(f'  {chunk},')
      lines.append('};\n')
      lines.append(f'static const unsigned int {array_name}_len = {len(data)};\n')

    else:  # data_type == 'float'
      # Array di float
      lines.append(f'static const float {array_name}[] = {{')
      float_values = [f'{val:.6f}f' for val in data]
      for i in range(0, len(float_values), 8):
        chunk = ', '.join(float_values[i:i+8])
        lines.append(f'  {chunk},')
      lines.append('};\n')
      lines.append(f'static const unsigned int {array_name}_len = {len(data)};\n')

    lines.append(f'#endif // {array_name.upper()}_H')

    # Salviamo il file header
    with open(header_filename, 'w') as hf:
      hf.write('\n'.join(lines))

    print(f"Generato header: {header_filename}")

# ESEMPIO DI UTILIZZO:
# 1) Avvia la funzione save_weights(model) => genera i file .bin
# 2) Poi esegui generate_headers() => genera i .h corrispondenti

if __name__ == '__main__':
  generate_headers()
