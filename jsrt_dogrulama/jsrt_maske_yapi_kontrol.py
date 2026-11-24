# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


from pathlib import Path
import cv2
import numpy as np

root = DATA / "JSRT" / "masks"
files = sorted(root.glob("*.tif"))

print("=== JSRT MASKE YAPISI ===")

for path in files:
    mask = read_image(
        str(path), cv2.IMREAD_UNCHANGED
    )

    if mask is None:
        print("Maske okunamadi:", path)
        continue
    binary = (mask > 0).astype(np.uint8)

    h, w = binary.shape

    left_pixels = np.count_nonzero(
        binary[:, :w // 2]
    )

    right_pixels = np.count_nonzero(
        binary[:, w // 2:]
    )

    print(
        path.name,
        "Sol yari:", left_pixels,
        "Sag yari:", right_pixels
    )
print("\nToplam kontrol edilen maske:", len(files))
