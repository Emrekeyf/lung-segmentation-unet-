# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


import cv2
import numpy as np
import torch
from pathlib import Path

from lungs_segmentation.pre_trained_models import create_model
import lungs_segmentation.inference as inference

# Sonuc klasoru
output = OUTPUTS / "modern_sonuclar"
output.mkdir(exist_ok=True)

# GPU varsa kullan
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Kullanilan cihaz:", device)

# Egitilmis modeli yukle
model = create_model("resnet34")
model = model.to(device)
model.eval()

# Iki rontgeni isle
for filename in ["rontgen_512.png", "stego.png"]:
    print("Isleniyor:", filename)

    if not (DATA / filename).exists():
        print("Dosya bulunamadi:", filename)
        continue

    with torch.no_grad():
        image, masks = model_inference(
            model, DATA / filename, 0.2
        )

    # Sag ve sol akciger maskeleri
    mask1 = np.asarray(masks[0])
    mask2 = np.asarray(masks[1])

    combined = np.logical_or(
        mask1 > 0, mask2 > 0
    ).astype(np.uint8) * 255

    original = read_image(
        DATA / filename, cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        raise ValueError("Goruntu okunamadi.")

    # Maskeyi orijinal boyuta getir
    combined = cv2.resize(
        combined,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Maskelenmis rontgen
    roi = cv2.bitwise_and(
        original, original, mask=combined
    )

    # Yesil maske gorunumu
    overlay = cv2.cvtColor(
        original, cv2.COLOR_GRAY2BGR
    )

    green = overlay.copy()
    green[combined > 0] = (0, 255, 0)

    overlay = cv2.addWeighted(
        overlay, 0.6, green, 0.4, 0
    )

    name = Path(filename).stem

    write_image(
        str(output / f"{name}_unet_maske.png"),
        combined
    )
    write_image(
        str(output / f"{name}_unet_roi.png"),
        roi
    )
    write_image(
        str(output / f"{name}_unet_overlay.png"),
        overlay
    )

    print("Tamamlandi:", filename)

print("Tum sonuclar modern_sonuclar klasorunde.")
