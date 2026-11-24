# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


import cv2
import numpy as np
import torch
import csv

from pathlib import Path
from lungs_segmentation.pre_trained_models import create_model
import lungs_segmentation.inference as inference

ROOT = DATA / "test_goruntuleri"
OUT = OUTPUTS / "toplu_sonuclar"
OUT.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Kullanilan cihaz:", device)

model = create_model("resnet34")
model = model.to(device)
model.eval()

rows = []
extensions = {".png", ".jpg", ".jpeg", ".bmp"}

for group in ["COVID", "Normal"]:
    folder = ROOT / group

    if not folder.exists():
        print("Klasor bulunamadi:", folder)
        continue

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    print(f"\n{group}: {len(images)} goruntu")

    group_out = OUT / group
    group_out.mkdir(exist_ok=True)

    for index, path in enumerate(images, 1):
        print(f"[{index}/{len(images)}] {path.name}")

        try:
            original = read_image(
                str(path), cv2.IMREAD_GRAYSCALE
            )

            if original is None:
                raise ValueError("Goruntu okunamadi")

            # Klasik on isleme
            median = cv2.medianBlur(original, 5)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            enhanced = clahe.apply(median)

            # Klasik Otsu segmentasyonu
            _, otsu = cv2.threshold(
                enhanced, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # Modern U-Net segmentasyonu
            with torch.no_grad():
                _, masks = model_inference(
                    model, str(path), 0.2
                )

            mask1 = np.asarray(masks[0]) > 0
            mask2 = np.asarray(masks[1]) > 0

            unet = np.logical_or(
                mask1, mask2
            ).astype(np.uint8) * 255

            unet = cv2.resize(
                unet,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            # Iki maske arasindaki ortusme
            a = otsu > 0
            b = unet > 0

            intersection = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            total = a.sum() + b.sum()

            dice = (
                2 * intersection / total
                if total else 1.0
            )

            iou = (
                intersection / union
                if union else 1.0
            )

            # Gorsel karsilastirma
            panels = []

            for title, image in [
                ("Orijinal", original),
                ("Median CLAHE", enhanced),
                ("Otsu", otsu),
                ("U-Net", unet)
            ]:
                panel = cv2.cvtColor(
                    image, cv2.COLOR_GRAY2BGR
                )

                cv2.putText(
                    panel, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )

                panels.append(panel)

            comparison = np.hstack(panels)

            write_image(
                str(group_out / f"{path.stem}_karsilastirma.png"),
                comparison
            )

            write_image(
                str(group_out / f"{path.stem}_unet_maske.png"),
                unet
            )

            rows.append({
                "Grup": group,
                "Dosya": path.name,
                "Dice": round(float(dice), 6),
                "IoU": round(float(iou), 6),
                "Otsu_Alan_Yuzdesi": round(
                    100 * float(a.mean()), 2
                ),
                "UNet_Alan_Yuzdesi": round(
                    100 * float(b.mean()), 2
                )
            })

        except Exception as e:
            print("HATA:", path.name, str(e))

# CSV kaydet
csv_path = OUT / "sonuclar.csv"

with open(
    csv_path, "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Grup", "Dosya", "Dice", "IoU",
            "Otsu_Alan_Yuzdesi", "UNet_Alan_Yuzdesi"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

# Grup ortalamalari
print("\n=== TOPLU ANALIZ SONUCLARI ===")

for group in ["COVID", "Normal"]:
    subset = [r for r in rows if r["Grup"] == group]

    if subset:
        mean_dice = np.mean([r["Dice"] for r in subset])
        mean_iou = np.mean([r["IoU"] for r in subset])

        print(
            f"{group}: {len(subset)} goruntu | "
            f"Dice={mean_dice:.4f} | "
            f"IoU={mean_iou:.4f}"
        )

print("\nToplam basarili:", len(rows))
print("Sonuclar:", csv_path)
