from xarray import DataArray
import numpy as np
from typing import Tuple, Optional, Any
from xarray_ome_ngff.v04.multiscale import (
    coords_to_transforms,
    multiscale_metadata,
    transforms_to_coords,
)

from pydantic_ome_ngff.v04.axis import Axis
from pydantic_ome_ngff.v04.multiscale import MultiscaleMetadata, Dataset
from pydantic_ome_ngff.v04.transform import (
    VectorScale,
    VectorTranslation,
)
import pint

ureg = pint.UnitRegistry()


def create_coord(
    shape: int,
    dim: str,
    units: Optional[str],
    scale: float,
    translate: float,
):
    return DataArray(
        (np.arange(shape) * scale) + translate,
        dims=(dim,),
        attrs={"units": units},
    )


def create_array(
    shape: Tuple[int, ...],
    dims: Tuple[str, ...],
    units: Tuple[Optional[str], ...],
    scale: Tuple[float, ...],
    translate: Tuple[float, ...],
    **kwargs: Any,
):
    """
    Create a dataarray with a shape and coordinates
    defined by the parameters axes, units, types, scale, translate.
    """
    coords = []
    for dim, unit, shp, scle, trns in zip(dims, units, shape, scale, translate):
        coords.append(
            create_coord(shape=shp, dim=dim, units=unit, scale=scle, translate=trns)
        )

    return DataArray(np.zeros(shape), coords=coords, **kwargs)


def test_ome_ngff_from_arrays():
    axes = ("z", "y", "x")
    units = ("meter", "meter", "meter")
    translate = (0, -8, 10)
    scale = (1.0, 1.0, 10.0)
    shape = (16,) * 3
    data = create_array(shape, axes, units, scale, translate)
    coarsen_kwargs = {**{dim: 2 for dim in axes}, "boundary": "trim"}
    multi = [data, data.coarsen(**coarsen_kwargs).mean()]
    multi.append(multi[-1].coarsen(**coarsen_kwargs).mean())
    array_paths = [f"s{idx}" for idx in range(len(multi))]
    axes, transforms = tuple(
        zip(*(coords_to_transforms(tuple(m.coords.values())) for m in multi))
    )
    multiscale_meta = multiscale_metadata(
        multi, array_paths=array_paths, name="foo"
    ).model_dump()
    expected_meta = MultiscaleMetadata(
        name="foo",
        datasets=[
            Dataset(path=array_paths[idx], coordinateTransformations=transforms[idx])
            for idx, m in enumerate(multi)
        ],
        axes=axes[0],
        coordinateTransformations=[VectorScale(scale=[1, 1, 1])],
    ).model_dump()

    assert multiscale_meta == expected_meta


def test_create_coords():
    shape = (3, 3)
    axes = [
        Axis(name="a", unit="meter", type="space"),
        Axis(name="b", unit="kilometer", type="space"),
    ]

    transforms = [
        VectorScale(scale=[1, 0.5]),
        VectorTranslation(translation=[1, 2]),
    ]

    coords = transforms_to_coords(axes, transforms, shape)
    assert coords[0].equals(
        DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=("a",),
            attrs={"units": "meter"},
        )
    )

    assert coords[1].equals(
        DataArray(
            np.array([2.0, 2.5, 3.0]),
            dims=("b",),
            attrs={"units": "kilometer"},
        )
    )


def test_create_axes_transforms():
    shape = (10, 10, 10)
    dims = ("z", "y", "x")
    units = ("meter", "nanometer", "kilometer")
    scale = (1, 2, 3)
    translate = (-1, 2, 0)

    array = create_array(
        shape=shape,
        dims=dims,
        units=units,
        scale=scale,
        translate=translate,
    )

    axes, transforms = coords_to_transforms(tuple(array.coords.values()))
    scale_tx, translation_tx = transforms

    for idx, ax in enumerate(axes):
        assert ax.unit == units[idx]
        assert ax.name == dims[idx]
        assert scale_tx.scale[idx] == scale[idx]
        assert translation_tx.translation[idx] == translate[idx]

    axes, transforms = coords_to_transforms(array.coords.values())
    scale_tx, translation_tx = transforms

    for idx, ax in enumerate(axes):
        assert ax.unit == units[idx]


def test_normalize_units():
    shape = (10, 10, 10)
    dims = ("z", "y", "x")
    units = ("m", "nm", "km")
    scale = (1, 2, 3)
    translate = (-1, 2, 0)

    array = create_array(
        shape=shape,
        dims=dims,
        units=units,
        scale=scale,
        translate=translate,
    )

    axes, _ = coords_to_transforms(array.coords.values())
    assert all(
        ax.unit == ureg.get_name(d, case_sensitive=True) for ax, d in zip(axes, units)
    )

    axes, _ = coords_to_transforms(array.coords.values(), normalize_units=False)
    assert all(ax.unit == d for ax, d in zip(axes, units))


def test_infer_axis_type():
    shape = (
        1,
        10,
        10,
        10,
    )
    dims = ("c", "t", "z", "y", "x")
    units = (None, "second", "meter", "nanometer", "kilometer")
    scale = (1, 1, 1, 2, 3)
    translate = (0, 0, -1, 2, 0)

    array = create_array(
        shape=shape,
        dims=dims,
        units=units,
        scale=scale,
        translate=translate,
    )

    axes, _ = coords_to_transforms(array.coords.values())
    for ax in axes:
        base_unit = ureg.get_base_units(ax.unit)[-1]
        if base_unit == ureg["meter"]:
            assert ax.type == "space"
        elif base_unit == ureg["second"]:
            assert ax.type == "time"
