import numpy as np

# Palette RGB utilisée pour mapper chaque classe remappée (0..7)
COLORS = np.array([
    [  0,   0,   0],   # 0 - background
    [128,  64, 128],   # 1 - road
    [244,  35, 232],   # 2 - sidewalk
    [ 70,  70,  70],   # 3 - building
    [102, 102, 156],   # 4 - construction
    [190, 153, 153],   # 5 - object
    [107, 142,  35],   # 6 - vegetation
    [  0,   0, 142],   # 7 - vehicle
], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convertit un masque de labels 2D en image RGB basée sur la palette."""
    # Vérification structurale du masque
    if mask.ndim != 2:
        raise ValueError(f"Le masque doit être 2D (H, W). Forme reçue : {mask.shape}")

    # Vérification que les labels correspondent aux indices définis
    if mask.min() < 0 or mask.max() >= len(COLORS):
        raise ValueError(
            f"Les valeurs du masque doivent être dans [0, {len(COLORS)-1}]. "
            f"Min={mask.min()}, Max={mask.max()}"
        )

    # Application directe de la palette (indexing vectorisé)
    return COLORS[mask]
