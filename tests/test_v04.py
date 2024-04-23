from __future__ import annotations

from dataclasses import dataclass
import zarr
import pytest
from xarray import DataArray
import numpy as np
from typing import Literal, Optional, Any
from xarray_ome_ngff.array_wrap import (
    DaskArrayWrapper,
    DaskArrayWrapperSpec,
    ZarrArrayWrapper,
    ZarrArrayWrapperSpec,
    resolve_wrapper,
)
from xarray_ome_ngff.core import CoordinateAttrs, ureg
from xarray_ome_ngff.v04.multiscale import (
    create_group,
    read_array,
    read_group,
    transforms_from_coords,
    model_group,
    multiscale_metadata,
    coords_from_transforms,
)
from zarr.storage import FSStore, BaseStore
from numcodecs import Zstd
from pydantic_ome_ngff.v04.axis import Axis
from pydantic_ome_ngff.v04.multiscale import MultiscaleMetadata, Dataset, Group
from pydantic_ome_ngff.v04.transform import (
    VectorScale,
    VectorTranslation,
)

try:
    import dask.array as da

    has_dask = True
except ImportError:
    has_dask = False

skip_no_dask = pytest.mark.skipif(not has_dask, reason="Dask not installed")


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
        attrs=CoordinateAttrs(units=units).model_dump(),
    )


def create_array(
    *,
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    units: tuple[str | None, ...],
    scale: tuple[float, ...],
    translate: tuple[float, ...],
    **kwargs: Any,
):
    """
    Create a `DataArray` with a shape and coordinates
    defined by the parameters axes, units, types, scale, translate.
    """
    coords = []
    for dim, unit, shp, scle, trns in zip(dims, units, shape, scale, translate):
        coords.append(
            create_coord(shape=shp, dim=dim, units=unit, scale=scle, translate=trns)
        )

    return DataArray(np.zeros(shape), coords=coords, **kwargs)


@pytest.fixture(scope="function")
def store(request, tmpdir):
    param: Literal["memory", "fsstore", "nested_directory"] = request.param

    if param == "memory":
        return zarr.MemoryStore()
    elif param == "fsstore":
        return FSStore(str(tmpdir))
    elif param == "nested_directory":
        return zarr.NestedDirectoryStore(str(tmpdir))


@dataclass
class PyramidRequest:
    shape: tuple[int, ...]
    dims: tuple[str, ...] | Literal["auto"] = "auto"
    units: tuple[str, ...] | Literal["auto"] = "auto"
    scale: tuple[float, ...] | Literal["auto"] = "auto"
    translate: tuple[float, ...] | Literal["auto"] = "auto"


@pytest.fixture(scope="function")
def pyramid(request) -> tuple[DataArray, DataArray, DataArray]:
    """
    Create a collection of DataArrays that represent a multiscale pyramid
    """
    param: PyramidRequest = request.param
    shape = param.shape
    if param.dims == "auto":
        dims = tuple(map(str, range(len(shape))))
    else:
        dims = param.dims

    if param.units == "auto":
        units = ("meter",) * len(shape)
    else:
        units = param.units

    if param.scale == "auto":
        scale = (1,) * len(shape)
    else:
        scale = param.scale

    if param.translate == "auto":
        translate = (0,) * len(shape)
    else:
        translate = param.translate

    data = create_array(
        shape=shape, dims=dims, units=units, scale=scale, translate=translate
    )

    coarsen_kwargs = {**{dim: 2 for dim in data.dims}, "boundary": "trim"}
    multi = (data, data.coarsen(**coarsen_kwargs).mean())
    multi += (multi[-1].coarsen(**coarsen_kwargs).mean(),)
    return multi


@pytest.mark.parametrize(
    "pyramid",
    [
        PyramidRequest(shape=(16,) * 3, units=("nanometer", "nm", "nanometer")),
    ],
    indirect=["pyramid"],
)
def test_make_pyramid(pyramid: tuple[DataArray, DataArray, DataArray]):
    assert len(pyramid) == 3
    assert all(isinstance(x, DataArray) for x in pyramid)


@pytest.mark.parametrize(
    "pyramid",
    [
        PyramidRequest(
            shape=(16,) * 3,
            scale=(1.0, 1.2, 0.5),
            translate=(0.0, 1.1, 0.0),
            dims=("z", "y", "x"),
        ),
        PyramidRequest(
            shape=(16,) * 4,
            units=("second", "nanometer", "nanometer", "nanometer"),
            scale=(2.0, 3.0, 5, 1.5),
        ),
    ],
    indirect=["pyramid"],
)
def test_multiscale_metadata(pyramid: tuple[DataArray, DataArray, DataArray]):
    array_paths = [f"s{idx}" for idx in range(len(pyramid))]
    multi_dict = dict(zip(array_paths, pyramid))
    axes, transforms = tuple(zip(*(transforms_from_coords(m.coords) for m in pyramid)))

    multiscale_meta = multiscale_metadata(multi_dict, name="foo").model_dump()

    expected_meta = MultiscaleMetadata(
        name="foo",
        datasets=tuple(
            Dataset(path=array_paths[idx], coordinateTransformations=transforms[idx])
            for idx, m in enumerate(pyramid)
        ),
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
        VectorScale(scale=(1, 0.5)),
        VectorTranslation(translation=(1, 2)),
    ]

    coords = coords_from_transforms(axes=axes, transforms=transforms, shape=shape)
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


@pytest.mark.parametrize(
    "pyramid",
    [
        PyramidRequest(
            shape=(16,) * 3,
            scale=(1.0, 1.2, 0.5),
            translate=(0.0, 1.1, 0.0),
            dims=("z", "y", "x"),
        ),
    ],
    indirect=["pyramid"],
)
def test_axes_consistent_dims(pyramid: tuple[DataArray, DataArray, DataArray]):
    """
    Test that creating a multiscale group from a collection of DataArray with inconsistent
    dimensions raises an exception.
    """

    # mutate the first array so that it has different dimensions from the rest
    p0 = pyramid[0]
    new_dims = tuple(str(k) + "_x" for k in p0.coords)
    pyramid_mutated = (
        DataArray(p0.data, dims=new_dims, coords=tuple(p0.coords.values())),
        *pyramid[1:],
    )
    assert pyramid_mutated[0].dims != p0.dims
    msg = (
        "Got 2 unique axes from `arrays` which means that their dimensions "
        "and / or coordinates are incompatible."
    )
    with pytest.raises(ValueError, match=msg):
        model_group(arrays=dict(zip(("s0", "s1", "s2"), pyramid_mutated)))


@pytest.mark.parametrize(
    "pyramid",
    [
        PyramidRequest(
            shape=(16,) * 3,
            scale=(1.0, 1.2, 0.5),
            translate=(0.0, 1.1, 0.0),
            dims=("z", "y", "x"),
            units=("meter", "meter", "meter"),
        ),
    ],
    indirect=["pyramid"],
)
def test_axes_consistent_units(pyramid: tuple[DataArray, DataArray, DataArray]):
    """
    Test that creating a multiscale group from a collection of DataArray with inconsistent
    units raises an exception.
    """

    # mutate the first array so that it has different units from the rest
    old_coords = tuple(pyramid[0].coords.items())
    for key, value in old_coords:
        value.attrs["units"] = "second"
        pyramid[0].coords[key] = value

    msg = (
        "Got 2 unique axes from `arrays` which means that their dimensions "
        "and / or coordinates are incompatible."
    )
    with pytest.raises(ValueError, match=msg):
        model_group(arrays=dict(zip(("s0", "s1", "s2"), pyramid)))


@pytest.mark.parametrize("normalize_units", [True, False])
@pytest.mark.parametrize("infer_axis_type", [True, False])
def test_create_axes_transforms(normalize_units: bool, infer_axis_type: bool) -> None:
    """
    Test that `Axis` and `coordinateTransformations` objects can be created from
    `DataArray` coordinates.
    """

    shape = (3, 10, 10, 10)
    dims = ("c", "t", "y", "x")
    scale = (1, 1, 2, 3)
    translate = (0, -1, 2, 0)

    if normalize_units:
        units_in = (None, "s", "nm", "km")
        units_expected = (
            None,
            *tuple(ureg.get_name(u, case_sensitive=True) for u in units_in[1:]),
        )
    else:
        units_in = (None, "second", "nanometer", "kilometer")
        units_expected = units_in

    if infer_axis_type:
        axis_type_expected = (None, "time", "space", "space")
    else:
        axis_type_expected = (None,) * len(shape)

    array = create_array(
        shape=shape,
        dims=dims,
        units=units_in,
        scale=scale,
        translate=translate,
    )

    axes, transforms = transforms_from_coords(
        array.coords, normalize_units=normalize_units, infer_axis_type=infer_axis_type
    )

    scale_tx, translation_tx = transforms

    for idx, ax in enumerate(axes):
        assert ax.unit == units_expected[idx]
        assert ax.name == dims[idx]
        assert ax.type == axis_type_expected[idx]
        assert scale_tx.scale[idx] == scale[idx]
        assert translation_tx.translation[idx] == translate[idx]


@pytest.mark.parametrize(
    "store", ["memory", "fsstore", "nested_directory"], indirect=["store"]
)
@pytest.mark.parametrize("paths", [("s0", "s1", "s2"), ("foo/s0", "foo/s1", "foo/s2")])
@pytest.mark.parametrize(
    "pyramid",
    [
        PyramidRequest(
            shape=(16,) * 3,
            scale=(1.0, 1.2, 0.5),
            translate=(0.0, 1.1, 0.0),
            dims=("z", "y", "x"),
        ),
        PyramidRequest(
            shape=(16,) * 4,
            units=("second", "nanometer", "nanometer", "nanometer"),
            scale=(2.0, 3.0, 5, 1.5),
        ),
    ],
    indirect=["pyramid"],
)
@pytest.mark.parametrize(
    "array_wrapper",
    (
        ZarrArrayWrapper(),
        {"name": "zarr_array", "config": {}},
        pytest.param(DaskArrayWrapper(chunks=10), marks=skip_no_dask),
        pytest.param(
            {"name": "dask_array", "config": {"chunks": 5}}, marks=skip_no_dask
        ),
    ),
)
@pytest.mark.parametrize("chunks", ("auto", 10))
@pytest.mark.parametrize("compressor", (None, Zstd(3)))
@pytest.mark.parametrize("fill_value", (0, 1))
def test_read_create_group(
    store: BaseStore,
    paths: tuple[str, str, str],
    pyramid: tuple[DataArray, DataArray, DataArray],
    array_wrapper: (
        ZarrArrayWrapper
        | DaskArrayWrapper
        | DaskArrayWrapperSpec
        | ZarrArrayWrapperSpec
    ),
    chunks: int | Literal["auto"],
    compressor: None | Zstd,
    fill_value: Literal[0, 1],
) -> None:

    # write some values to the arrays
    pyramid[0][:] = 1
    pyramid[1][:] = 2
    pyramid[2][:] = 3

    arrays = dict(zip(paths, pyramid))
    axes, transforms = tuple(zip(*(transforms_from_coords(m.coords) for m in pyramid)))
    if isinstance(chunks, int):
        _chunks = (chunks,) * pyramid[0].ndim
    else:
        _chunks = chunks

    expected_group_model = Group.from_arrays(
        arrays=tuple(arrays.values()),
        paths=tuple(arrays.keys()),
        axes=axes[0],
        scales=[t[0].scale for t in transforms],
        translations=[t[1].translation for t in transforms],
        chunks=_chunks,
        compressor=compressor,
        fill_value=fill_value,
    )

    observed_group_model = model_group(
        arrays=arrays, chunks=_chunks, fill_value=fill_value, compressor=compressor
    )

    assert observed_group_model == expected_group_model

    # now test reconstructing our original arrays
    zarr_group = create_group(
        store=store, path="test", arrays=arrays, transform_precision=8
    )

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
        assert expected_array.attrs == observed_array.attrs
        assert observed_array.equals(expected_array)


@pytest.mark.parametrize(
    "store", ["memory", "fsstore", "nested_directory"], indirect=["store"]
)
@pytest.mark.parametrize("paths", [("s0", "s1", "s2"), ("foo/s0", "foo/s1", "foo/s2")])
@pytest.mark.parametrize(
    "pyramid",
    [
        PyramidRequest(
            shape=(16,) * 3,
            scale=(1.0, 1.2, 0.5),
            translate=(0.0, 1.1, 0.0),
            dims=("z", "y", "x"),
        ),
        PyramidRequest(
            shape=(16,) * 4,
            units=("second", "nanometer", "nanometer", "nanometer"),
            scale=(2.0, 3.0, 5, 1.5),
        ),
    ],
    indirect=["pyramid"],
)
@pytest.mark.parametrize(
    "array_wrapper",
    (
        ZarrArrayWrapper(),
        {"name": "zarr_array", "config": {}},
        pytest.param(DaskArrayWrapper(chunks=10), marks=skip_no_dask),
        pytest.param(
            {"name": "dask_array", "config": {"chunks": 5}}, marks=skip_no_dask
        ),
    ),
)
def test_multiscale_to_array(
    store: BaseStore,
    paths: tuple[str, str, str],
    pyramid: tuple[DataArray, DataArray, DataArray],
    array_wrapper: (
        ZarrArrayWrapper
        | DaskArrayWrapper
        | DaskArrayWrapperSpec
        | ZarrArrayWrapperSpec
    ),
) -> None:

    # write some values to the arrays
    pyramid[0][:] = 1
    pyramid[1][:] = 2
    pyramid[2][:] = 3

    if isinstance(array_wrapper, dict):
        array_wrapper_parsed = resolve_wrapper(array_wrapper)
    else:
        array_wrapper_parsed = array_wrapper

    arrays = dict(zip(paths, pyramid))
    zarr_group = create_group(
        store=store, path="test", arrays=arrays, transform_precision=8
    )
    for name, arr_expc in arrays.items():
        arr_obs = read_array(zarr_group[name], array_wrapper=array_wrapper)
        if isinstance(array_wrapper_parsed, ZarrArrayWrapper):
            assert isinstance(arr_obs.data, np.ndarray)
        elif isinstance(array_wrapper_parsed, DaskArrayWrapper):
            assert isinstance(arr_obs.data, da.Array)
            assert (
                arr_obs.data.chunksize
                == arr_obs.data.rechunk(array_wrapper_parsed.chunks).chunksize
            )

        for coord_name in arr_expc.coords:
            assert arr_obs.coords[coord_name].equals(arr_expc.coords[coord_name])
