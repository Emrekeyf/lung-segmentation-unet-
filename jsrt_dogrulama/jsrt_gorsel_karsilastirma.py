# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from lungs_segmentation.pre_trained_models import create_model
from lungs_segmentation.inference import inference

root = DATA / "JSRT"
output_dir = OUTPUTS / "JSRT" / "gorseller"
output_dir.mkdir(parents=True, exist_ok=True)

samples = [
    "JPCNN047",
    "JPCNN080",
    "JPCLN107",
    "JPCNN071",
    "JPCLN039"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Kullanilan cihaz:", device)

model = create_model("resnet34").to(device)
model.eval()

def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return 2 * intersection / total if total else 1.0

for name in samples:
    image_path = root / "jpg" / f"{name}.jpg"
    mask_path = root / "yeni_maskeler" / f"{name}.png"

    image = read_image(str(image_path), cv2.IMREAD_GRAYSCALE)
    gt = read_image(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or gt is None:
        print("Dosya okunamadi:", name)
        continue

    h, w = image.shape

    gt = cv2.resize(
        gt, (w, h),
        interpolation=cv2.INTER_NEAREST
    )
    gt = (gt >= 128).astype(np.uint8)

    # Klasik yontem
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

    # U-Net
    with torch.no_grad():
        _, masks = model_inference(model, str(image_path), 0.2)

    unet = np.logical_or(np.asarray(masks[0]) > 0, np.asarray(masks[1]) > 0).astype(np.uint8)
    unet = cv2.resize(
        unet, (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    otsu_dice = dice_score(otsu, gt)
    unet_dice = dice_score(unet, gt)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    panels = [
        (image, "Orijinal Rontgen"),
        (gt, "Gercek Maske"),
        (otsu, f"Otsu | Dice: {otsu_dice:.4f}"),
        (unet, f"U-Net | Dice: {unet_dice:.4f}")
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255 if title == "Orijinal Rontgen" else 1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        f"JSRT - {name} | Segmentasyon Karsilastirmasi",
        fontsize=14
    )
    plt.tight_layout()

    output_path = output_dir / f"{name}_karsilastirma.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(
        f"{name} tamam | "
        f"Otsu Dice: {otsu_dice:.4f} | "
        f"U-Net Dice: {unet_dice:.4f}"
    )

print("\nTum gorseller kaydedildi:", output_dir)
