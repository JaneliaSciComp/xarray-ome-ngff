from xarray_ome_ngff.registry import get_adapters
from xarray_ome_ngff.core import ngff_versions
import pytest


@pytest.mark.parametrize("version", [*ngff_versions, "latest"])
def test_registry(version: str):
    adapters = get_adapters(version)
    if adapters.ngff_version == "0.4":
        from xarray_ome_ngff.v04.multiscale import (
            multiscale_metadata,
            coords_to_transforms,
            transforms_to_coords,
        )

        assert adapters.multiscale_metadata == multiscale_metadata
        assert adapters.coords_to_transforms == coords_to_transforms
        assert adapters.transforms_to_coords == transforms_to_coords
