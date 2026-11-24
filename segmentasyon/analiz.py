# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

a = read_image(DATA / "rontgen_512.png", 0)
b = read_image(DATA / "stego.png", 0)

if a is None or b is None:
    raise FileNotFoundError("Rontgen dosyalari bulunamadi.")

if a.shape != b.shape:
    raise ValueError("Goruntu boyutlari ayni olmali.")

def isle(img):
    median = cv2.medianBlur(img, 5)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(median)

    _, otsu = cv2.threshold(
        clahe, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    sobel_x = cv2.Sobel(clahe, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(clahe, cv2.CV_64F, 0, 1)
    sobel = cv2.convertScaleAbs(
        cv2.magnitude(sobel_x, sobel_y)
    )

    canny = cv2.Canny(clahe, 50, 150)

    return {
        "Orijinal": img,
        "Median": median,
        "CLAHE": clahe,
        "Otsu": otsu,
        "Sobel": sobel,
        "Canny": canny
    }

def iou(x, y):
    x = x > 0
    y = y > 0
    union = np.logical_or(x, y).sum()
    if union == 0:
        return 1.0
    return np.logical_and(x, y).sum() / union

sonuc_a = isle(a)
sonuc_b = isle(b)

print("\n--- GORUNTU KARSILASTIRMA ---")

for ad in sonuc_a:
    x = sonuc_a[ad]
    y = sonuc_b[ad]

    psnr = cv2.PSNR(x, y)
    ssim_deger = ssim(x, y, data_range=255)

    print(f"\n{ad}")
    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim_deger:.6f}")

    if ad in ["Otsu", "Canny"]:
        print(f"IoU: {iou(x, y):.6f}")
