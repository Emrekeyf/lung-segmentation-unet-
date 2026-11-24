# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


from pathlib import Path
import cv2
import numpy as np

root = DATA / "JSRT"

names = ["JPCLN001", "JPCLN050", "JPCNN001"]

for name in names:
    xray = read_image(str(root / "jpg" / f"{name}.jpg"))
    mask = read_image(
        str(root / "yeni_maskeler" / f"{name}.png"),
        cv2.IMREAD_GRAYSCALE
    )

    if xray is None or mask is None:
        print(f"{name}: Dosya okunamadi")
        continue

    print(f"{name}: Rontgen {xray.shape[:2]}, Maske {mask.shape}")

    mask = cv2.resize(
        mask,
        (xray.shape[1], xray.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    binary = (mask > 0).astype(np.uint8)

    overlay = xray.copy()
    green = np.zeros_like(xray)
    green[:, :, 1] = 255

    area = binary.astype(bool)
    overlay[area] = cv2.addWeighted(
        xray, 0.6, green, 0.4, 0
    )[area]

    output = root / f"kontrol_{name}.png"
    write_image(str(output), overlay)

    print(f"Kaydedildi: {output}")

print("Kontrol tamamlandi.")
