#!/usr/bin/env python3

import os
import numpy as np
import shutil

def generate_headers(weights_dir='weights_16bit', headers_dir='../Core/Inc/weights_16bit/'):
    """
    Generates .h header files from binary files stored in `weights_dir`.
    Recognizes:
      - '_binarized.bin'  => bit-packed with 0/1 (uint8)
      - '_float16.bin'    => array of float16 (stored as uint16)
    Additionally, it first clears the `headers_dir` folder (if it exists) to avoid conflicts.
    """

    # Empty the header folder
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

        # Ignore files that do not end with '.bin'
        if not filename.endswith('.bin'):
            continue

        # Determine the data type based on the suffix
        if '_binarized.bin' in filename:
            data_type = 'binarized'
        elif '_float16.bin' in filename:
            data_type = 'float16'
        else:
            # Unknown type => skip (or warn)
            print(f"[generate_headers] WARNING: '{filename}' does not contain '_binarized' or '_float16'. Skipped.")
            continue

        # Generate a "base" name without the .bin extension
        base_name = os.path.splitext(filename)[0]
        # Replace any dots or special characters in the array name
        array_name = base_name.replace('.', '_')

        # Create the path for the header file
        header_filename = os.path.join(headers_dir, f'{base_name}.h')

        # === Read binary file ===
        if data_type == 'binarized':
            # Read data as uint8 (bit-packed 0/1)
            with open(file_path, 'rb') as f:
                raw = f.read()
            data = np.frombuffer(raw, dtype=np.uint8)

        else:  # float16
            data = np.fromfile(file_path, dtype=np.float16)

        # === Create header content ===
        lines = []
        lines.append(f'// Automatically generated header for {filename}')
        lines.append(f'#ifndef {array_name.upper()}_H')
        lines.append(f'#define {array_name.upper()}_H\n')

        if data_type == 'binarized':
            lines.append('// Binarized data (bit-packed 0/1)')
            lines.append(f'static const unsigned char {array_name}[] = {{')
            hex_values = [f'0x{val:02X}' for val in data]
            # Print 12 values per line
            for i in range(0, len(hex_values), 12):
                chunk = ', '.join(hex_values[i:i+12])
                lines.append(f'  {chunk},')
            lines.append('};\n')
            lines.append(f'static const unsigned int {array_name}_len = {len(data)};\n')

        else:  # float16
            # Preserve the original bits by converting to uint16
            uint16_data = data.view(np.uint16)
            lines.append('// Float16 data stored as uint16_t array (preserves the bit pattern of float16)')
            lines.append('#include <stdint.h>')
            lines.append(f'static const uint16_t {array_name}[] = {{')
            hex_values = [f'0x{val:04X}' for val in uint16_data]
            # Print 8 values per line
            for i in range(0, len(hex_values), 8):
                chunk = ', '.join(hex_values[i:i+8])
                lines.append(f'  {chunk},')
            lines.append('};\n')
            lines.append(f'static const unsigned int {array_name}_len = {len(uint16_data)};\n')

        lines.append(f'#endif // {array_name.upper()}_H')

        # === Save the header file ===
        with open(header_filename, 'w') as hf:
            hf.write('\n'.join(lines))

        print(f"[generate_headers] Header generated: {header_filename}")


if __name__ == '__main__':
    generate_headers()
