# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


import cv2
import numpy as np

# Orijinal rontgen
img = read_image(DATA / "rontgen_512.png", 0)

# U-Net tarafindan uretilen maske
unet = read_image(
    OUTPUTS / "modern_sonuclar/rontgen_512_unet_maske.png", 0
)

if img is None or unet is None:
    raise FileNotFoundError("Gerekli goruntuler bulunamadi.")

# Klasik Otsu segmentasyonu
blur = cv2.medianBlur(img, 5)
_, otsu = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# U-Net maskesini ayni boyuta getir
unet = cv2.resize(
    unet, (img.shape[1], img.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# Karsilastirma panelleri
panels = [
    ("Orijinal Rontgen", img),
    ("Klasik Otsu", otsu),
    ("Modern U-Net", unet)
]

results = []

for title, image in panels:
    panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        panel, title, (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 255), 2
    )
    results.append(panel)

comparison = np.hstack(results)

write_image(
    OUTPUTS / "modern_sonuclar/otsu_unet_karsilastirma.png",
    comparison
)

print("Karsilastirma gorseli olusturuldu.")
