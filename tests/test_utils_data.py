from pathlib import Path
from src.utils.utils_data import list_available_ids

def test_list_available_ids(tmp_path):
    (tmp_path / "img1.png").touch()
    (tmp_path / "img2.png").touch()

    ids = list_available_ids(tmp_path)
    assert len(ids) == 2
