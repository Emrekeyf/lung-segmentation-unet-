"""JSRT tek tarafli TIFF / iki tarafli PNG referanslarinin kokenini kontrol etmeye yardimci olur."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_utils import DATA, read_image
import cv2
root = DATA / 'JSRT'
images = sorted((root / 'jpg').glob('*.jpg'))
raw = sorted((root / 'masks').glob('*.tif'))
processed = sorted((root / 'yeni_maskeler').glob('*.png'))
print('Rontgen:', len(images), 'Ham TIFF:', len(raw), 'Islenmis PNG:', len(processed))
print('UYARI: Sayi ve goruntu benzerligi uzman referansinin kokenini kanitlamaz.')
for path in processed[:5]:
    mask = read_image(path, cv2.IMREAD_GRAYSCALE)
    if mask is None: print('Okunamadi:', path.name); continue
    print(path.name, 'boyut:', mask.shape, 'piksel min/max:', int(mask.min()), int(mask.max()))
