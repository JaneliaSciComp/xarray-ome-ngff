import pytest

from xarray_ome_ngff.core import ngff_versions
from xarray_ome_ngff.registry import get_adapters


@pytest.mark.parametrize("version", [*ngff_versions, "latest"])
def test_registry(version: str):
    adapters = get_adapters(version)
    if adapters.ngff_version == "0.4":
        from xarray_ome_ngff.v04.multiscales import (
            coords_to_transforms,
            multiscale_metadata,
            transforms_to_coords,
        )

        assert adapters.multiscale_metadata == multiscale_metadata
        assert adapters.coords_to_transforms == coords_to_transforms
        assert adapters.transforms_to_coords == transforms_to_coords

    elif adapters.ngff_version in ("0.5-dev", "latest"):
        from xarray_ome_ngff.latest.multiscales import (
            coords_to_transforms,
            multiscale_metadata,
            transforms_to_coords,
        )

        assert adapters.multiscale_metadata == multiscale_metadata
        assert adapters.coords_to_transforms == coords_to_transforms
        assert adapters.transforms_to_coords == transforms_to_coords
