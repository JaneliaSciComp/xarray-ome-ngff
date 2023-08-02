from importlib.metadata import version as _version

__version__ = _version(__name__)

# ruff: noqa
from xarray_ome_ngff.registry import get_adapters
