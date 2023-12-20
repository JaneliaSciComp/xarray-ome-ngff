from __future__ import annotations

from typing import Any, Callable, NamedTuple

from xarray_ome_ngff.core import ngff_versions


class MetadataAdapters(NamedTuple):
    ngff_version: str
    multiscale_metadata: Callable[[Any], Any]
    transforms_to_coords: Callable[[Any], Any]
    coords_to_transforms: Callable[[Any], Any]


def get_adapters(version: str):
    if version == "0.4":
        from xarray_ome_ngff.v04.multiscales import (
            coords_to_transforms,
            multiscale_metadata,
            transforms_to_coords,
        )

        return MetadataAdapters(
            ngff_version="0.4",
            multiscale_metadata=multiscale_metadata,
            transforms_to_coords=transforms_to_coords,
            coords_to_transforms=coords_to_transforms,
        )

    if version in ("0.5-dev", "latest"):
        from xarray_ome_ngff.latest.multiscales import (
            coords_to_transforms,
            multiscale_metadata,
            transforms_to_coords,
        )

        return MetadataAdapters(
            ngff_version="0.5-dev",
            multiscale_metadata=multiscale_metadata,
            transforms_to_coords=transforms_to_coords,
            coords_to_transforms=coords_to_transforms,
        )

    msg = f"Got version={version}, but this is not one of the supported versions: {ngff_versions}"
    raise TypeError(msg)
