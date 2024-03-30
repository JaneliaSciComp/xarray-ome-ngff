from __future__ import annotations
from typing import TYPE_CHECKING
from importlib.metadata import version as _version

from xarray_ome_ngff.core import NGFF_VERSIONS

if TYPE_CHECKING:
    from typing import Literal
from xarray import DataArray
import zarr

__version__ = _version(__name__)

from xarray_ome_ngff.v04 import multiscale as multiscale_v04


def read_multiscale_group(
    group: zarr.Group, ngff_version: Literal["0.4"], **kwargs
) -> dict[str, DataArray]:
    if ngff_version not in NGFF_VERSIONS:
        raise ValueError(f"Unsupported NGFF version: {ngff_version}")
    if ngff_version == "0.4":
        return multiscale_v04.read_group(group, **kwargs)


def read_multiscale_array(
    array: zarr.Array, ngff_version: Literal["0.4"], **kwargs
) -> DataArray:
    if ngff_version not in NGFF_VERSIONS:
        raise ValueError(f"Unsupported NGFF version: {ngff_version}")
    if ngff_version == "0.4":
        return multiscale_v04.read_array(array, **kwargs)
