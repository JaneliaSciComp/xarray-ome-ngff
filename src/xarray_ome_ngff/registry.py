from typing import NamedTuple, Callable, Any
from xarray_ome_ngff.core import ngff_versions


class MetadataAdapters(NamedTuple):
    ngff_version: str
    multiscale_metadata: Callable[[Any], Any]
    transforms_to_coords: Callable[[Any], Any]
    coords_to_transforms: Callable[[Any], Any]


def get_adapters(version: str):
    if version == "0.4":
        from xarray_ome_ngff.v04.multiscales import (
            multiscale_metadata,
            transforms_to_coords,
            coords_to_transforms,
        )

        return MetadataAdapters(
            ngff_version="0.4",
            multiscale_metadata=multiscale_metadata,
            transforms_to_coords=transforms_to_coords,
            coords_to_transforms=coords_to_transforms,
        )

    elif version == "0.5-dev" or version == "latest":
        from xarray_ome_ngff.latest.multiscales import (
            multiscale_metadata,
            transforms_to_coords,
            coords_to_transforms,
        )

        return MetadataAdapters(
            ngff_version="0.5-dev",
            multiscale_metadata=multiscale_metadata,
            transforms_to_coords=transforms_to_coords,
            coords_to_transforms=coords_to_transforms,
        )

    else:
        raise ValueError(
            f"""
                Got version={version}, but this is not one of the supported versions:
                {ngff_versions}
                    """
        )
