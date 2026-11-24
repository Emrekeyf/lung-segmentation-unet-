# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

from lungs_segmentation.pre_trained_models import create_model
from lungs_segmentation.inference import inference

root = DATA / "JSRT"
image_dir = root / "jpg"
mask_dir = root / "yeni_maskeler"
output_dir = OUTPUTS / "JSRT"
output_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Kullanilan cihaz:", device)

model = create_model("resnet34")
model = model.to(device)
model.eval()

def metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    union = np.logical_or(pred, gt).sum()

    dice = (2 * intersection / (pred_sum + gt_sum)
            if pred_sum + gt_sum else 1.0)
    iou = intersection / union if union else 1.0

    return float(dice), float(iou)

results = []
errors = []

images = sorted(image_dir.glob("*.jpg"))

for index, image_path in enumerate(images, 1):
    try:
        name = image_path.stem

        image = read_image(
            str(image_path), cv2.IMREAD_GRAYSCALE
        )
        gt = read_image(
            str(mask_dir / f"{name}.png"),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or gt is None:
            raise ValueError("Rontgen veya maske okunamadi")

        h, w = image.shape

        gt = cv2.resize(
            gt, (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        gt = (gt >= 128).astype(np.uint8)

        # Klasik yontem: Median + CLAHE + Otsu
        median = cv2.medianBlur(image, 5)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(median)

        _, otsu = cv2.threshold(
            enhanced, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        otsu = (otsu > 0).astype(np.uint8)

        # Modern yontem: Onceden egitilmis U-Net

        with torch.no_grad():
            _, masks = model_inference(model, str(image_path), 0.2)

        # Iki akciger maskesini birlestir
        unet = np.logical_or(np.asarray(masks[0]) > 0, np.asarray(masks[1]) > 0).astype(np.uint8)

        unet = cv2.resize(
            unet,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )










         
        otsu_dice, otsu_iou = metrics(otsu, gt)
        unet_dice, unet_iou = metrics(unet, gt)

        results.append({
            "Goruntu": name,
            "Otsu_Dice": otsu_dice,
            "Otsu_IoU": otsu_iou,
            "UNet_Dice": unet_dice,
            "UNet_IoU": unet_iou
        })

        print(f"[{index}/{len(images)}] {name} tamam")

    except Exception as e:
        errors.append((image_path.name, str(e)))
        print(f"HATA: {image_path.name}: {e}")

df = pd.DataFrame(results)

if not df.empty:
    csv_path = output_dir / "jsrt_dogruluk_sonuclari.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n=== JSRT SONUCLARI ===")
    print("Basarili:", len(results))
    print("Hatali:", len(errors))
    print("\nOrtalama degerler:")
    print(df.drop(columns=["Goruntu"]).mean().round(4))
    print("\nCSV:", csv_path)
else:
    print("Basarili analiz yok.")

if errors:
    print("\nHatalar:")
    for filename, message in errors:
        print(filename, message)
