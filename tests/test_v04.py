import zarr
import pytest
from xarray import DataArray
import numpy as np
from typing import Optional, Any
from xarray_ome_ngff.v04.multiscale import (
    DaskArrayWrapper,
    DaskArrayWrapperSpec,
    ZarrArrayWrapper,
    ZarrArrayWrapperSpec,
    create_group,
    read_group,
    transforms_from_coords,
    model_group,
    multiscale_metadata,
    resolve_wrapper,
    coords_from_transforms,
)
import dask.array as da
from pydantic_ome_ngff.v04.axis import Axis
from pydantic_ome_ngff.v04.multiscale import MultiscaleMetadata, Dataset, Group
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
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    units: tuple[str | None, ...],
    scale: tuple[float, ...],
    translate: tuple[float, ...],
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


def test_multiscale_metadata():
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
    multi_dict = dict(zip(array_paths, multi))
    axes, transforms = tuple(zip(*(transforms_from_coords(m.coords) for m in multi)))

    multiscale_meta = multiscale_metadata(multi_dict, name="foo").model_dump()

    expected_meta = MultiscaleMetadata(
        name="foo",
        datasets=[
            Dataset(path=array_paths[idx], coordinateTransformations=transforms[idx])
            for idx, m in enumerate(multi)
        ],
        axes=axes[0],
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

    coords = coords_from_transforms(axes, transforms, shape)
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

    axes, transforms = transforms_from_coords(array.coords)
    scale_tx, translation_tx = transforms

    for idx, ax in enumerate(axes):
        assert ax.unit == units[idx]
        assert ax.name == dims[idx]
        assert scale_tx.scale[idx] == scale[idx]
        assert translation_tx.translation[idx] == translate[idx]

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

    axes, _ = transforms_from_coords(array.coords)
    assert all(
        ax.unit == ureg.get_name(d, case_sensitive=True) for ax, d in zip(axes, units)
    )

    axes, _ = transforms_from_coords(array.coords, normalize_units=False)
    assert all(ax.unit == d for ax, d in zip(axes, units))


def test_infer_axis_type() -> None:
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

    axes, _ = transforms_from_coords(array.coords)
    for ax in axes:
        base_unit = ureg.get_base_units(ax.unit)[-1]
        if base_unit == ureg["meter"]:
            assert ax.type == "space"
        elif base_unit == ureg["second"]:
            assert ax.type == "time"


@pytest.mark.parametrize(
    "array_wrapper",
    (
        ZarrArrayWrapper(),
        DaskArrayWrapper(chunks=10),
        {"name": "zarr_array", "config": {}},
        {"name": "dask_array", "config": {"chunks": 10}},
    ),
)
@pytest.mark.parametrize("ndim", (2, 3, 4))
def test_read_create_group(
    array_wrapper: (
        ZarrArrayWrapper
        | DaskArrayWrapper
        | DaskArrayWrapperSpec
        | ZarrArrayWrapperSpec
    ),
    ndim: int,
) -> None:

    store = zarr.MemoryStore()

    if ndim == 2:
        axes = ("z", "y")
        units = ("nanometer",) * ndim
    elif ndim == 3:
        axes = ("z", "y", "x")
        units = ("nanometer",) * ndim
    elif ndim == 4:
        axes = ("t", "z", "y", "x")
        units = ("second",) + ("nanometer",) * 3
    axis_types = ["space" if unit == "nanometer" else "time" for unit in units]
    axis_objects = [
        Axis(name=ax, type=axis_types[idx], unit=units[idx])
        for idx, ax in enumerate(axes)
    ]
    arrays = {}
    base_scale = [1.0, 2.0, 3.0, 4.0][:ndim]
    base_trans = [0.0, 1.5, 2.5, 3.5][:ndim]
    s1_scale = [2 * x for x in base_scale]
    s1_trans = [t + s / 2 for t, s in zip(base_trans, base_scale)]
    arrays["s0"] = create_array(
        shape=(10,) * ndim,
        units=units,
        dims=axes,
        scale=base_scale,
        translate=base_trans,
        name="name_s0",
    )

    arrays["s1"] = create_array(
        shape=(5,) * ndim,
        units=units,
        dims=axes,
        scale=s1_scale,
        translate=s1_trans,
        name="name_s1",
    )

    # write some values to the arrays
    arrays["s0"][:] = 1
    arrays["s1"][:] = 2

    expected_group_model = Group.from_arrays(
        arrays=arrays.values(),
        paths=arrays.keys(),
        axes=axis_objects,
        scales=[base_scale, s1_scale],
        translations=[base_trans, s1_trans],
    )

    observed_group_model = model_group(arrays=arrays)

    assert observed_group_model == expected_group_model

    # now test reconstructing our original arrays
    zarr_group = create_group(store, path="test", arrays=arrays)

    # write the array data
    for path, arr in arrays.items():
        zarr_group[path][:] = arr.data

    observed_arrays = read_group(zarr_group, array_wrapper=array_wrapper)

    if isinstance(array_wrapper, dict):
        array_wrapper_parsed = resolve_wrapper(array_wrapper)
    else:
        array_wrapper_parsed = array_wrapper

    for path, expected_array in arrays.items():
        observed_array = observed_arrays[path]
        if isinstance(array_wrapper_parsed, ZarrArrayWrapper):
            assert isinstance(observed_array.data, np.ndarray)
        elif isinstance(array_wrapper_parsed, DaskArrayWrapper):
            assert isinstance(observed_array.data, da.Array)
            assert (
                observed_array.data.chunksize
                == observed_array.data.rechunk(array_wrapper_parsed.chunks).chunksize
            )
        assert observed_array.equals(expected_array)


def multiscale_to_array():
    group_model = Group.from_arrays(
        (np.zeros((10,)), np.zeros((5,))),
        ("s0", "s1"),
        axes=(Axis(name="x", type="space", unit="nanometer")),
    )
