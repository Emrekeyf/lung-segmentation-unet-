# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


import cv2
import numpy as np

orijinal = read_image(
    OUTPUTS / "modern_sonuclar/rontgen_512_unet_maske.png",
    cv2.IMREAD_GRAYSCALE
)

stego = read_image(
    OUTPUTS / "modern_sonuclar/stego_unet_maske.png",
    cv2.IMREAD_GRAYSCALE
)

if orijinal is None or stego is None:
    raise FileNotFoundError(
        "U-Net maske dosyalari bulunamadi."
    )

if orijinal.shape != stego.shape:
    raise ValueError(
        "Maskelerin boyutlari ayni olmali."
    )

a = orijinal > 0
b = stego > 0

kesisim = np.logical_and(a, b).sum()
birlesim = np.logical_or(a, b).sum()

dice = 2 * kesisim / (a.sum() + b.sum()) if (a.sum() + b.sum()) else 1.0
iou = kesisim / birlesim if birlesim else 1.0

fark = np.logical_xor(a, b).astype(np.uint8) * 255

write_image(
    OUTPUTS / "modern_sonuclar/unet_maske_farklari.png",
    fark
)

print("=== U-NET MASKE KARSILASTIRMASI ===")
print(f"Dice: {dice:.6f}")
print(f"IoU: {iou:.6f}")
print(f"Dice (%): {dice * 100:.2f}")
print(f"IoU (%): {iou * 100:.2f}")
print("Fark goruntusu kaydedildi.")
