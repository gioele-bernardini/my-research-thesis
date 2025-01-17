#!/usr/bin/env python3

import os
import numpy as np
import shutil

def generate_headers(weights_dir='weights_16bit', headers_dir='headers'):
    """
    Genera file header .h a partire dai file binari salvati in `weights_dir`.
    Riconosce:
      - '_binarized.bin'  => bit-pack con 0/1
      - '_float16.bin'    => array di float16 (salvato come uint16)
    Inoltre, svuota prima la cartella `headers_dir` (se esiste) per evitare conflitti.
    """

    # Svuota la cartella degli header, se esiste
    if os.path.exists(headers_dir):
        for item in os.listdir(headers_dir):
            item_path = os.path.join(headers_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        os.makedirs(headers_dir)

    for filename in os.listdir(weights_dir):
        file_path = os.path.join(weights_dir, filename)

        # Ignora i file che non terminano con '.bin'
        if not filename.endswith('.bin'):
            continue

        # Determina il tipo di dati dal suffisso
        if '_binarized.bin' in filename:
            data_type = 'binarized'
        elif '_float16.bin' in filename:
            data_type = 'float16'
        else:
            # Non sappiamo che tipo sia, gestiscilo come preferisci
            print(f"Attenzione: {filename} non contiene né 'binarized' né 'float16'. Skippato.")
            continue

        # Genera un nome "base" senza estensione .bin
        base_name = os.path.splitext(filename)[0]  
        # Evita punti o caratteri speciali nel nome dell'array
        array_name = base_name.replace('.', '_')   

        # Crea il path per il file header
        header_filename = os.path.join(headers_dir, f'{base_name}.h')

        # === Lettura file ===
        if data_type == 'binarized':
            # Legge i dati in uint8 (bit-pack 0/1)
            with open(file_path, 'rb') as f:
                raw = f.read()
            data = np.frombuffer(raw, dtype=np.uint8)

        else:  # float16
            data = np.fromfile(file_path, dtype=np.float16)

        # === Creazione contenuto header ===
        lines = []
        lines.append(f'// Header generato automaticamente per {filename}')
        lines.append(f'#ifndef {array_name.upper()}_H')
        lines.append(f'#define {array_name.upper()}_H\n')

        if data_type == 'binarized':
            lines.append(f'// Dati binarizzati (bit-pack 0/1)')
            lines.append(f'static const unsigned char {array_name}[] = {{')
            hex_values = [f'0x{val:02X}' for val in data]
            # Stampa 12 valori per riga
            for i in range(0, len(hex_values), 12):
                chunk = ', '.join(hex_values[i:i+12])
                lines.append(f'  {chunk},')
            lines.append('};\n')
            lines.append(f'static const unsigned int {array_name}_len = {len(data)};\n')

        else:  # data_type == 'float16'
            # Manteniamo i bit originali del float16 convertendolo in uint16
            uint16_data = data.view(np.uint16)
            lines.append(f'// Dati float16 salvati come array di uint16_t (mantiene il bit pattern dei float16)')
            lines.append('#include <stdint.h>')
            lines.append(f'static const uint16_t {array_name}[] = {{')
            hex_values = [f'0x{val:04X}' for val in uint16_data]
            # Stampa 8 valori per riga
            for i in range(0, len(hex_values), 8):
                chunk = ', '.join(hex_values[i:i+8])
                lines.append(f'  {chunk},')
            lines.append('};\n')
            lines.append(f'static const unsigned int {array_name}_len = {len(uint16_data)};\n')

        lines.append(f'#endif // {array_name.upper()}_H')

        # === Salvataggio file header ===
        with open(header_filename, 'w') as hf:
            hf.write('\n'.join(lines))

        print(f"Generato header: {header_filename}")


# ESEMPIO DI UTILIZZO:
# 1) Avvia la funzione save_weights_16bit(model) => genera i file .bin (binarizzati o float16)
# 2) Poi esegui generate_headers() => genera i .h corrispondenti
if __name__ == '__main__':
    generate_headers()
