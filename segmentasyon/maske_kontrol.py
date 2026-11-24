# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


from pathlib import Path

root = DATA / "MontgomerySet"

images = root / "CXR_png"
left = root / "ManualMask" / "leftMask"
right = root / "ManualMask" / "rightMask"

image_files = sorted(images.glob("*.png"))

print("=== MONTGOMERY DOSYA KONTROLU ===")
print("Rontgen sayisi:", len(image_files))

missing_left = []
missing_right = []

for image in image_files:
    if not (left / image.name).exists():
        missing_left.append(image.name)

    if not (right / image.name).exists():
        missing_right.append(image.name)

print("Eslesmeyen sol maske:", len(missing_left))
print("Eslesmeyen sag maske:", len(missing_right))

print("\nOrnek rontgen:", image_files[0].name if image_files else "Veri bulunamadi")

print("\nSol maske ornekleri:")
for p in sorted(left.glob("*.png"))[:3]:
    print(p.name)

print("\nSag maske ornekleri:")
for p in sorted(right.glob("*.png"))[:3]:
    print(p.name)

if not missing_left and not missing_right:
    print("\nTum dosya adlari eslesiyor.")
else:
    print("\nDosya adlandirmasi kontrol edilmeli.")
