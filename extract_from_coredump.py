#!/usr/bin/env python3
"""
Extract neural network weights from a Python core dump.
Usage: python extract_from_coredump.py /path/to/coredump

This searches for numpy array patterns in the raw memory dump.
"""

import sys
import struct
import numpy as np
import re

def find_numpy_arrays(data, min_size=10, max_size=1000):
    """
    Search for numpy array-like float64 sequences in binary data.
    Neural network weights are typically float64 arrays with values in [-1, 1].
    """
    arrays_found = []

    # Search for sequences of float64 that look like weights
    i = 0
    while i < len(data) - 8:
        # Try to read as float64
        try:
            vals = []
            j = i
            while j < len(data) - 8 and len(vals) < max_size:
                val = struct.unpack('d', data[j:j+8])[0]
                # Neural net weights are typically in reasonable range
                if -10 < val < 10 and not np.isnan(val) and not np.isinf(val):
                    vals.append(val)
                    j += 8
                else:
                    break

            if len(vals) >= min_size:
                # Check if values look like weights (mostly in [-1, 1])
                arr = np.array(vals)
                if np.abs(arr).mean() < 2 and np.abs(arr).max() < 10:
                    arrays_found.append((i, arr))
                    i = j  # Skip past this array
                    continue
        except:
            pass
        i += 8

    return arrays_found

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_from_coredump.py /path/to/coredump")
        sys.exit(1)

    corefile = sys.argv[1]
    print(f"Reading {corefile}...")

    # Read in chunks to handle large files
    chunk_size = 100 * 1024 * 1024  # 100MB chunks
    all_arrays = []

    with open(corefile, 'rb') as f:
        offset = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            print(f"Searching offset {offset // (1024*1024)}MB...")
            arrays = find_numpy_arrays(chunk)
            for pos, arr in arrays:
                all_arrays.append((offset + pos, arr))
            offset += len(chunk)

    print(f"\nFound {len(all_arrays)} potential weight arrays")

    # Group by size (neural net layers have specific sizes)
    by_size = {}
    for pos, arr in all_arrays:
        size = len(arr)
        if size not in by_size:
            by_size[size] = []
        by_size[size].append((pos, arr))

    print("\nArrays by size:")
    for size in sorted(by_size.keys()):
        print(f"  Size {size}: {len(by_size[size])} arrays")

    # Save all found arrays
    output = 'extracted_weights.npz'
    save_dict = {f'array_{i}': arr for i, (pos, arr) in enumerate(all_arrays)}
    np.savez(output, **save_dict)
    print(f"\nSaved to {output}")

if __name__ == '__main__':
    main()
