import os, tempfile, pytest
from veritas import VeritasDB


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    return VeritasDB(str(path))
