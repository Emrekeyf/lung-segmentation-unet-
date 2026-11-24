# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


from pathlib import Path
import cv2
import numpy as np

root = DATA / "JSRT"

images = sorted((root / "jpg").glob("*.jpg"))
masks = root / "masks"

missing = []
unreadable = []

print("=== JSRT VERI KUMESI KONTROLU ===")
print("Rontgen sayisi:", len(images))
print("Maske sayisi:", len(list(masks.glob("*.tif"))))

for image_path in images:
    mask_path = masks / (image_path.stem + ".tif")

    if not mask_path.exists():
        missing.append(image_path.name)
        continue

    image = read_image(
        str(image_path), cv2.IMREAD_GRAYSCALE
    )
    mask = read_image(
        str(mask_path), cv2.IMREAD_UNCHANGED
    )

    if image is None or mask is None:
        unreadable.append(image_path.name)

print("Eksik maske:", len(missing))
print("Okunamayan dosya:", len(unreadable))

if images:
    sample = images[0]
    mask_path = masks / (sample.stem + ".tif")

    image = read_image(
        str(sample), cv2.IMREAD_GRAYSCALE
    )
    mask = read_image(
        str(mask_path), cv2.IMREAD_UNCHANGED
    )

    print("\nOrnek rontgen:", sample.name)
    print("Ornek maske:", mask_path.name)

    if image is not None and mask is not None:
        print("Rontgen boyutu:", image.shape)
        print("Maske boyutu:", mask.shape)
        print("Maske veri tipi:", mask.dtype)
        print("Maske min/max:", mask.min(), mask.max())
        print("Maske benzersiz deger sayisi:",
              len(np.unique(mask)))

if not missing and not unreadable:
    print("\nTum goruntuler ve maskeler okunabiliyor.")
else:
    print("\nKontrol edilmesi gereken dosyalar var.")
    print("Eksik ornekler:", missing[:5])
    print("Okunamayan ornekler:", unreadable[:5])
