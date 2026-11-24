# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


import cv2
import numpy as np
from pathlib import Path

output = OUTPUTS / "segmentasyon_sonuclari"
output.mkdir(exist_ok=True)

for filename in ["rontgen_512.png", "stego.png"]:
    img = read_image(DATA / filename, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Dosya okunamadi: {filename}")
        continue

    # 1. Gurultu azaltma
    median = cv2.medianBlur(img, 5)

    # 2. Kontrast iyilestirme
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(median)

    # 3. Koyu bolgeleri aday olarak secme
    _, mask = cv2.threshold(
        clahe, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4. Morfolojik temizleme
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    # 5. En buyuk iki bolgeyi secme
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours, key=cv2.contourArea,
        reverse=True
    )[:2]

    roi_mask = np.zeros_like(mask)
    cv2.drawContours(
        roi_mask, contours, -1, 255, -1
    )

    # 6. ROI goruntusunu olusturma
    roi = cv2.bitwise_and(
        img, img, mask=roi_mask
    )

    # 7. Maskeyi orijinal goruntuye bindirme
    overlay = cv2.cvtColor(
        img, cv2.COLOR_GRAY2BGR
    )
    overlay[roi_mask > 0] = (
        0.5 * overlay[roi_mask > 0] +
        0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    name = Path(filename).stem

    write_image(str(output / f"{name}_maske.png"), roi_mask)
    write_image(str(output / f"{name}_roi.png"), roi)
    write_image(str(output / f"{name}_overlay.png"), overlay)

    print(f"{filename}: Islem tamamlandi.")

print("Tum sonuclar segmentasyon_sonuclari klasorunde.")
