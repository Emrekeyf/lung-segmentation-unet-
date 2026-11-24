"""Portable file access for educational lung-segmentation scripts."""
from pathlib import Path
import tempfile
import shutil
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data"
OUTPUTS = PROJECT_ROOT / "outputs"

def read_image(path, flags=cv2.IMREAD_COLOR):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    except (OSError, ValueError, cv2.error):
        return None

def write_image(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, array)
    if not ok:
        raise OSError(f"Gorsel kodlanamadi: {path}")
    encoded.tofile(str(path))
    return True

def model_inference(model, image_path, threshold=0.2):
    """Use temporary ASCII filename because upstream inference uses cv2.imread.

    Returns the upstream inference result unchanged; does not train a model.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Rontgen bulunamadi: {image_path}")
    import lungs_segmentation.inference as module
    # Public Documents is generally writable and has an ASCII Windows path.
    # If unavailable, fall back to OS temporary directory; an error will expose
    # any remaining path/permission problem rather than silently skipping it.
    import os
    if os.name == 'nt':
        public = Path(os.environ.get('PUBLIC', r'C:\Users\Public')) / 'Documents'
        base = public if public.is_dir() else Path(tempfile.gettempdir())
    else:
        base = Path(tempfile.gettempdir())
    with tempfile.TemporaryDirectory(prefix='lungseg_', dir=str(base)) as td:
        tmp = Path(td) / ('input' + image_path.suffix.lower())
        shutil.copyfile(image_path, tmp)
        return module.inference(model, str(tmp), threshold)
