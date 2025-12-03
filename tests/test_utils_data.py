import sys
from pathlib import Path

# Ajout de la racine du projet au PYTHONPATH afin de rendre "src" importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.utils_data import list_available_ids


def test_list_available_ids(tmp_path):
    """Test de la détection des IDs d’images selon la convention Cityscapes."""
    
    # Création de fichiers respectant la nomenclature attendue
    (tmp_path / "foo_leftImg8bit.png").touch()
    (tmp_path / "bar_leftImg8bit.png").touch()

    ids = list_available_ids(tmp_path)

    # Vérifie que les suffixes sont correctement retirés
    assert set(ids) == {"foo", "bar"}
