from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def list_available_ids(images_dir: str | Path) -> List[str]:
    """Retourne la liste des IDs d'images disponibles à partir des noms Cityscapes."""
    images_dir = Path(images_dir)
    ids: List[str] = []

    # Parcours des images respectant la nomenclature Cityscapes
    for img_path in sorted(images_dir.glob("*_leftImg8bit.png")):
        name = img_path.name
        # Extraction de l'ID sans suffixe
        image_id = name.replace("_leftImg8bit.png", "")
        ids.append(image_id)

    return ids


def load_image_and_mask(
    image_id: str,
    images_dir: str | Path,
    masks_dir: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Charge une image RGB et son masque Cityscapes (labelIds) à partir d'un ID."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    # Construction des chemins selon la convention Cityscapes
    img_name = f"{image_id}_leftImg8bit.png"
    mask_name = f"{image_id}_gtFine_labelIds.png"

    img_path = images_dir / img_name
    mask_path = masks_dir / mask_name

    # Validation de l'existence des fichiers nécessaires
    if not img_path.exists():
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Masque introuvable : {mask_path}")

    # Chargement de l'image (OpenCV lit en BGR)
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Échec de lecture image : {img_path}")
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Chargement du masque tel quel (étiquettes labelIds)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Échec de lecture masque : {mask_path}")

    # Simplification en canal unique si nécessaire
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return image_rgb, mask.astype(np.uint8)
