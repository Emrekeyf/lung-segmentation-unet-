# Repo-relative paths and Unicode-safe image I/O.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_utils import DATA, OUTPUTS, read_image, write_image, model_inference


import numpy as np
import torch

from lungs_segmentation.pre_trained_models import create_model
from lungs_segmentation.inference import inference

image_path = DATA / "JSRT" / "jpg" / "JPCLN001.jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = create_model("resnet34")
model = model.to(device)
model.eval()

with torch.no_grad():
    _, masks = model_inference(model, image_path, 0.2)

print("Cikti tipi:", type(masks))
print("Maske sayisi:", len(masks))

for i, mask in enumerate(masks):
    arr = np.asarray(mask)
    print(
        f"Maske {i}:",
        "Boyut =", arr.shape,
        "Veri tipi =", arr.dtype,
        "Min =", arr.min(),
        "Max =", arr.max()
    )
