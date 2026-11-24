# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


from pathlib import Path
import cv2
import numpy as np

root = DATA / "JSRT"

image_path = root / "jpg" / "JPCLN001.jpg"
mask_path = root / "masks" / "JPCLN001.tif"

image = read_image(
    str(image_path), cv2.IMREAD_GRAYSCALE
)

mask = read_image(
    str(mask_path), cv2.IMREAD_UNCHANGED
)

if image is None or mask is None:
    raise RuntimeError("Goruntu veya maske okunamadi.")

# Maskeyi rontgen boyutuna indir
h, w = image.shape

mask = cv2.resize(
    mask, (w, h),
    interpolation=cv2.INTER_NEAREST
)

mask = (mask > 0).astype(np.uint8) * 255

# Renkli bindirme olustur
overlay = cv2.cvtColor(
    image, cv2.COLOR_GRAY2BGR
)

green = np.zeros_like(overlay)
green[:, :, 1] = mask

overlay = cv2.addWeighted(
    overlay, 0.7, green, 0.3, 0
)

# Goruntuleri yan yana getir
panels = []

for title, img in [
    ("Orijinal", image),
    ("Gercek Maske", mask),
    ("Maske Bindirme", overlay)
]:
    if len(img.shape) == 2:
        img = cv2.cvtColor(
            img, cv2.COLOR_GRAY2BGR
        )

    img = cv2.resize(img, (512, 512))

    cv2.putText(
        img, title, (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 255), 2
    )

    panels.append(img)

result = np.hstack(panels)

output = OUTPUTS / "JSRT" / "jsrt_gorsel_kontrol.png"
write_image(str(output), result)

print("Gorsel olusturuldu:", output.resolve())
print("Maske doluluk orani:",
      round(np.mean(mask > 0) * 100, 2), "%")
