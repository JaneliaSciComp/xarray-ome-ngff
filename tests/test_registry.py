from xarray_ome_ngff.registry import get_adaptors
from xarray_ome_ngff.core import ngff_versions
import pytest


@pytest.mark.parametrize("version", [*ngff_versions, "latest"])
def test_registry(version: str):
    adaptors = get_adaptors(version)
    if adaptors.ngff_version == "0.4":
        from xarray_ome_ngff.v04.multiscales import (
            multiscale_metadata,
            coords_to_transforms,
            transforms_to_coords,
        )

        assert adaptors.multiscale_metadata == multiscale_metadata
        assert adaptors.coords_to_transforms == coords_to_transforms
        assert adaptors.transforms_to_coords == transforms_to_coords

    elif adaptors.ngff_version == "0.5-dev" or adaptors.ngff_version == "latest":
        from xarray_ome_ngff.latest.multiscales import (
            multiscale_metadata,
            coords_to_transforms,
            transforms_to_coords,
        )

        assert adaptors.multiscale_metadata == multiscale_metadata
        assert adaptors.coords_to_transforms == coords_to_transforms
        assert adaptors.transforms_to_coords == transforms_to_coords
