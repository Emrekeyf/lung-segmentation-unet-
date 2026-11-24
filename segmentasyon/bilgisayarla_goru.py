# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference

from pathlib import Path
import cv2
import numpy as np

ROOT = DATA
OUT = OUTPUTS / 'sonuclar'
OUT.mkdir(exist_ok=True)

def process(filename):
    path = ROOT / filename
    img = read_image(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'HATA: Goruntu okunamadi: {path}')
        return
    median = cv2.medianBlur(img, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(median)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sobel_x = cv2.Sobel(clahe, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(clahe, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))
    canny = cv2.Canny(clahe, 50, 150)
    images = {'00_orijinal': img, '01_median': median, '02_clahe': clahe,
              '03_otsu': otsu, '04_sobel': sobel, '05_canny': canny}
    stem = Path(filename).stem
    for label, arr in images.items():
        write_image(str(OUT / f'{stem}_{label}.png'), arr)
    top = np.hstack([cv2.cvtColor(images[k], cv2.COLOR_GRAY2BGR) for k in list(images)[:3]])
    bottom = np.hstack([cv2.cvtColor(images[k], cv2.COLOR_GRAY2BGR) for k in list(images)[3:]])
    combined = np.vstack([top, bottom])
    write_image(str(OUT / f'{stem}_karsilastirma.png'), combined)
    print(f'Tamamlandi: {filename} -> {OUT}')

if __name__ == '__main__':
    process('rontgen_512.png')
    if (ROOT / 'stego.png').exists():
        process('stego.png')
    print('Not: Otsu maskesi dogrudan akciger segmentasyonu anlamina gelmez.')
