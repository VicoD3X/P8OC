import sys
from pathlib import Path

# Ajout de la racine du projet au PYTHONPATH pour que "src" soit importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.utils_data import list_available_ids


def test_list_available_ids(tmp_path):
    (tmp_path / "img1.png").touch()
    (tmp_path / "img2.png").touch()

    ids = list_available_ids(tmp_path)
    assert len(ids) == 2
