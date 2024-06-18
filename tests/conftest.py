import pytest
from zarr.storage import DirectoryStore, MemoryStore, FSStore


@pytest.fixture(scope="function")
def store(request, tmpdir):
    if request.param == "memory_store":
        return MemoryStore()
    elif request.param == "directory_store":
        return DirectoryStore(str(tmpdir))
    elif request.param == "fsstore_local":
        return FSStore(str(tmpdir))
    raise AssertionError
