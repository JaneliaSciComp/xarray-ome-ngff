from xarray import DataArray
import numpy as np
from typing import Tuple, Optional, Any
from xarray_ome_ngff.v05.multiscales import (
    create_transforms,
    create_multiscale,
    create_coords,
)

from pydantic_ome_ngff.v05.axes import Axis
from pydantic_ome_ngff.v05.multiscales import Multiscale, MultiscaleDataset
from pydantic_ome_ngff.v05.coordinateTransformations import (
    VectorScaleTransform,
    VectorTranslationTransform,
)
import pint

ureg = pint.UnitRegistry()


def create_coord(
    shape: int,
    dim: str,
    unit: Optional[str],
    type: Optional[str],
    scale: float,
    translate: float,
):
    return DataArray(
        (np.arange(shape) * scale) + translate,
        dims=(dim,),
        attrs={"unit": unit, "type": type},
    )


def create_array(
    shape: Tuple[int, ...],
    dims: Tuple[str, ...],
    units: Tuple[Optional[str], ...],
    types: Tuple[Optional[str], ...],
    scale: Tuple[float, ...],
    translate: Tuple[float, ...],
    **kwargs: Any,
):
    """
    Create a dataarray with a shape and coordinates
    defined by the parameters axes, units, types, scale, translate.
    """
    coords = []
    for dim, unit, shp, scle, trns, typ in zip(
        dims, units, shape, scale, translate, types
    ):
        coords.append(
            create_coord(
                shape=shp, dim=dim, unit=unit, type=typ, scale=scle, translate=trns
            )
        )

    return DataArray(np.zeros(shape), coords=coords, **kwargs)


def test_ome_ngff_from_arrays():
    axes = ("z", "y", "x")
    units = ("meter", "meter", "meter")
    types = ("space", "space", "space")
    translate = (0, -8, 10)
    scale = (1.0, 1.0, 10.0)
    shape = (16,) * 3
    data = create_array(shape, axes, units, types, scale, translate)
    coarsen_kwargs = {**{dim: 2 for dim in axes}, "boundary": "trim"}
    multi = [data, data.coarsen(**coarsen_kwargs).mean()]
    multi.append(multi[-1].coarsen(**coarsen_kwargs).mean())
    array_paths = [f"s{idx}" for idx in range(len(multi))]
    axes, transforms = tuple(zip(*(create_transforms(m) for m in multi)))
    multiscale_meta = create_multiscale(
        multi, array_paths=array_paths, name="foo"
    ).dict()
    expected_meta = Multiscale(
        name="foo",
        datasets=[
            MultiscaleDataset(
                path=array_paths[idx], coordinateTransformations=transforms[idx]
            )
            for idx, m in enumerate(multi)
        ],
        axes=axes[0],
        coordinateTransformations=[VectorScaleTransform(scale=[1, 1, 1])],
    ).dict()

    assert multiscale_meta == expected_meta


def test_create_coords():
    shape = (3, 3)
    axes = [
        Axis(name="a", unit="meter", type="space"),
        Axis(name="b", unit="kilometer", type="space"),
    ]

    transforms = [
        VectorScaleTransform(scale=[1, 0.5]),
        VectorTranslationTransform(translation=[1, 2]),
    ]

    coords = create_coords(axes, transforms, shape)
    assert coords[0].equals(
        DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=("a",),
            attrs={"unit": "meter", "type": "space"},
        )
    )

    assert coords[1].equals(
        DataArray(
            np.array([2.0, 2.5, 3.0]),
            dims=("b",),
            attrs={"unit": "kilometer", "type": "space"},
        )
    )


def test_create_axes_transforms():
    shape = (10, 10, 10)
    dims = ("z", "y", "x")
    units = ("meter", "nanometer", "kilometer")
    types = ("space",) * 3
    scale = (1, 2, 3)
    translate = (-1, 2, 0)

    array = create_array(
        shape=shape,
        dims=dims,
        units=units,
        types=types,
        scale=scale,
        translate=translate,
    )

    axes, transforms = create_transforms(array)
    scale_tx, translation_tx = transforms

    for idx, ax in enumerate(axes):
        assert ax.unit == units[idx]
        assert ax.type == types[idx]
        assert ax.name == dims[idx]
        assert scale_tx.scale[idx] == scale[idx]
        assert translation_tx.translation[idx] == translate[idx]

    # change 'unit' to 'units' in coordinate metadata
    for dim in array.dims:
        unit = array.coords[dim].attrs.pop("unit")
        array.coords[dim].attrs["units"] = unit

    axes, transforms = create_transforms(array)
    scale_tx, translation_tx = transforms

    for idx, ax in enumerate(axes):
        assert ax.unit == units[idx]


def test_normalize_units():
    shape = (10, 10, 10)
    dims = ("z", "y", "x")
    units = ("m", "nm", "km")
    types = ("space",) * 3
    scale = (1, 2, 3)
    translate = (-1, 2, 0)

    array = create_array(
        shape=shape,
        dims=dims,
        units=units,
        types=types,
        scale=scale,
        translate=translate,
    )

    axes, _ = create_transforms(array)
    assert all(
        ax.unit == ureg.get_name(d, case_sensitive=True) for ax, d in zip(axes, units)
    )

    axes, _ = create_transforms(array, normalize_units=False)
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
    types = (None, None, "space", None, "time")
    scale = (1, 1, 1, 2, 3)
    translate = (0, 0, -1, 2, 0)

    array = create_array(
        shape=shape,
        dims=dims,
        units=units,
        types=types,
        scale=scale,
        translate=translate,
    )

    axes, _ = create_transforms(array)
    for ax, typ in zip(axes, types):
        if typ is None:
            base_unit = ureg.get_base_units(ax.unit)[-1]
            if base_unit == ureg["meter"]:
                assert ax.type == "space"
            elif base_unit == ureg["second"]:
                assert ax.type == "time"
        else:
            assert ax.type == typ
