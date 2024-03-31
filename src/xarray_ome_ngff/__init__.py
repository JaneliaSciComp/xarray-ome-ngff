from __future__ import annotations
from typing import TYPE_CHECKING
from importlib.metadata import version as _version
from zarr.storage import BaseStore
from xarray_ome_ngff.array_wrap import DaskArrayWrapper, ZarrArrayWrapper
from xarray_ome_ngff.core import NGFF_VERSIONS

if TYPE_CHECKING:
    from typing import Literal
    from pydantic_ome_ngff.v04.multiscale import Group as MultiscaleGroupV04

from xarray import DataArray
import zarr
from xarray_ome_ngff.v04 import multiscale as multiscale_v04

__version__ = _version(__name__)


# todo: remove the need to specify the version for reading
def read_multiscale_group(
    group: zarr.Group, ngff_version: Literal["0.4"] = "0.4", **kwargs
) -> dict[str, DataArray]:
    if ngff_version not in NGFF_VERSIONS:
        raise ValueError(f"Unsupported NGFF version: {ngff_version}")
    if ngff_version == "0.4":
        return multiscale_v04.read_group(group, **kwargs)


# todo: remove the need to specify the version for reading
def read_multiscale_array(
    array: zarr.Array, ngff_version: Literal["0.4"] = "0.4", **kwargs
) -> DataArray:
    """
    Read a Zarr array as an `xarray.DataArray` with coordinates derived from ngff metadata
    """
    if ngff_version not in NGFF_VERSIONS:
        raise ValueError(f"Unsupported NGFF version: {ngff_version}")
    if ngff_version == "0.4":
        return multiscale_v04.read_array(array, **kwargs)


def model_multiscale_group(
    *,
    arrays: dict[str, DataArray],
    transform_precision: int | None = None,
    ngff_version: Literal["0.4"] = "0.4",
) -> MultiscaleGroupV04:
    """
    Create a `pydantic` model of a Zarr group with NGFF multiscale image metadata.
    """
    if ngff_version not in NGFF_VERSIONS:
        raise ValueError(f"Unsupported NGFF version: {ngff_version}")
    if ngff_version == "0.4":
        return multiscale_v04.model_group(
            arrays=arrays, transform_precision=transform_precision
        )


def create_multiscale_group(
    *,
    store: BaseStore,
    path: str,
    arrays: dict[str, DataArray],
    transform_precision: int | None = None,
    ngff_version: Literal["0.4"] = "0.4",
):
    """
    Create a Zarr group that complies with OME-NGFF multiscale metadata from a `dict` of
    `xarray.DataArray`.
    """
    if ngff_version not in NGFF_VERSIONS:
        raise ValueError(f"Unsupported NGFF version: {ngff_version}")
    if ngff_version == "0.4":
        return multiscale_v04.create_group(
            store=store,
            path=path,
            arrays=arrays,
            transform_precision=transform_precision,
        )


__all__ = [
    "read_multiscale_array",
    "read_multiscale_array",
    "create_multiscale_group",
    "model_multiscale_group",
    "DaskArrayWrapper",
    "ZarrArrayWrapper",
]
