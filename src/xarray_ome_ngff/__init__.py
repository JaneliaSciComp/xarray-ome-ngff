from importlib.metadata import version as _version
from xarray import DataArray

__version__ = _version(__name__)

from xarray_ome_ngff.v04.multiscale import create_group, read_group, read_array
