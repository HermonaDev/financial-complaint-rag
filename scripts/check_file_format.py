#!/usr/bin/env python3
"""Try to determine actual file format."""

import struct
from pathlib import Path

file_path = Path('data/raw/complaint_embeddings-001.parquet')
print(f"Checking file: {file_path}")
print(f"Size: {file_path.stat().st_size / 1024**3:.2f} GB")

# Read first 100 bytes
with open(file_path, 'rb') as f:
    header = f.read(100)
    
print("\nFirst 100 bytes (hex):")
for i in range(0, len(header), 16):
    hex_part = ' '.join(f'{b:02x}' for b in header[i:i+16])
    ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in header[i:i+16])
    print(f"{i:04x}: {hex_part:<48} {ascii_part}")

# Check for common file signatures
print("\nChecking for known file signatures:")

# Check for Parquet signature (should end with "PAR1")
if header[-4:] == b'PAR1':
    print("  ✓ Parquet signature found (PAR1)")
else:
    print("  ✗ Not a valid Parquet file (missing PAR1 footer)")

# Check for Arrow feather format
if header[:6] == b'ARROW1':
    print("  ✓ Arrow feather format detected")
else:
    print("  ✗ Not Arrow feather format")

# Check for CSV/TSV by looking for text
text_chars = sum(1 for b in header if 32 <= b < 127 or b in (9, 10, 13))
if text_chars > 80:  # Mostly text
    print("  ✓ Appears to be text-based (CSV, TSV, JSON, etc.)")
    # Try to decode as UTF-8
    try:
        decoded = header.decode('utf-8', errors='ignore')
        print(f"  First 100 chars as text: {decoded[:100]}")
    except:
        print("  Cannot decode as UTF-8")
else:
    print("  ✗ Not primarily text-based")

# Check for gzip compression
if header[:2] == b'\x1f\x8b':
    print("  ✓ gzip compressed file")
    
# Check for zip compression
if header[:4] == b'PK\x03\x04':
    print("  ✓ ZIP compressed file")

print("\nTrying to read as different formats...")

# Try to read as text file
try:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
    print(f"  As text, first line: {first_line[:100]}")
    print(f"  As text, second line: {second_line[:100]}")
except Exception as e:
    print(f"  Cannot read as text: {e}")

# Try to check if it's actually a large CSV with wrong extension
print("\nChecking file extension pattern...")
if 'complaint_embeddings' in str(file_path) and file_path.suffix == '.parquet':
    print("  File name suggests it should contain embeddings")
    print("  According to challenge doc, this should be pre-built embeddings")
    print("  But the file appears corrupted or in wrong format")
