# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


from pathlib import Path
import csv
import cv2
import numpy as np
import torch

from lungs_segmentation.pre_trained_models import create_model
import lungs_segmentation.inference as inference

ROOT = DATA / "MontgomerySet"
OUT = OUTPUTS / "MontgomerySet"
OUT.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Kullanilan cihaz:", device)

model = create_model("resnet34")
model = model.to(device)
model.eval()

def metrics(pred, gt):
    pred = pred > 0
    gt = gt > 0

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    total = pred.sum() + gt.sum()

    dice = 2 * intersection / total if total else 1.0
    iou = intersection / union if union else 1.0

    return float(dice), float(iou)

images = sorted((ROOT / "CXR_png").glob("*.png"))
results = []
failures = []

for index, path in enumerate(images, 1):
    print(f"[{index}/{len(images)}] {path.name}")

    try:
        original = read_image(
            str(path), cv2.IMREAD_GRAYSCALE
        )

        left = read_image(
            str(ROOT / "ManualMask" / "leftMask" / path.name),
            cv2.IMREAD_GRAYSCALE
        )

        right = read_image(
            str(ROOT / "ManualMask" / "rightMask" / path.name),
            cv2.IMREAD_GRAYSCALE
        )

        if original is None or left is None or right is None:
            raise ValueError("Goruntu veya referans maske okunamadi")

        size = (original.shape[1], original.shape[0])

        left = cv2.resize(
            left, size, interpolation=cv2.INTER_NEAREST
        )
        right = cv2.resize(
            right, size, interpolation=cv2.INTER_NEAREST
        )

        ground_truth = np.logical_or(
            left > 0, right > 0
        ).astype(np.uint8) * 255

        # Klasik Otsu
        median = cv2.medianBlur(original, 5)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(median)

        _, otsu = cv2.threshold(
            enhanced, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Modern U-Net
        with torch.no_grad():
            _, masks = model_inference(
                model, str(path), 0.2
            )

        unet = np.logical_or(
            np.asarray(masks[0]) > 0,
            np.asarray(masks[1]) > 0
        ).astype(np.uint8) * 255

        unet = cv2.resize(
            unet, size, interpolation=cv2.INTER_NEAREST
        )

        # Gercek referans maskeye gore olcum
        otsu_dice, otsu_iou = metrics(
            otsu, ground_truth
        )

        unet_dice, unet_iou = metrics(
            unet, ground_truth
        )

        results.append({
            "Dosya": path.name,
            "Otsu_Dice": otsu_dice,
            "Otsu_IoU": otsu_iou,
            "UNet_Dice": unet_dice,
            "UNet_IoU": unet_iou
        })

        # Ilk 5 goruntu icin gorsel kaydet
        if index <= 5:
            panels = []

            for title, image in [
                ("Orijinal", original),
                ("Gercek Maske", ground_truth),
                ("Otsu", otsu),
                ("U-Net", unet)
            ]:
                image_small = cv2.resize(
                    image, (512, 512),
                    interpolation=cv2.INTER_NEAREST
                )

                panel = cv2.cvtColor(
                    image_small, cv2.COLOR_GRAY2BGR
                )

                cv2.putText(
                    panel, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )

                panels.append(panel)

            write_image(
                str(OUT / f"{path.stem}_karsilastirma.png"),
                np.hstack(panels)
            )

    except Exception as e:
        failures.append((path.name, str(e)))
        print("HATA:", e)

# Sonuclari CSV olarak kaydet
with open(
    OUT / "gercek_dogruluk_sonuclari.csv",
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Dosya", "Otsu_Dice", "Otsu_IoU",
            "UNet_Dice", "UNet_IoU"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\n=== GERCEK SEGMENTASYON DOGRULUGU ===")
print("Basarili:", len(results))
print("Hatali:", len(failures))

if results:
    for key in [
        "Otsu_Dice", "Otsu_IoU",
        "UNet_Dice", "UNet_IoU"
    ]:
        avg = np.mean([r[key] for r in results])
        print(f"{key}: {avg:.4f} ({avg * 100:.2f}%)")

print("\nSonuclar referans_sonuclar klasorunde.")
