from xarray import DataArray
import numpy as np
import tempfile
import shutil
import atexit
import zarr
import ome_zarr
from ome_zarr.writer import write_image
from ome_zarr.reader import Reader
from ome_zarr.io import parse_url
import os

from xarray_ome_ngff.v05.multiscales import (
    AxesTransformsFromDataArray,
    MultiscaleGroupMetadata,
    OmeNgffVersion,
    MultiscaleDataset,
    MultiscaleMetadata,
)


def create_dataarray(shape, axes, units, scale, translate, **kwargs):
    """
    Create a dataarray with a shape and coordinates defined by the parameters axes, units, scale, translate.
    """
    coords = []
    for ax, u, sh, sc, tr in zip(axes, units, shape, scale, translate):
        coords.append(
            DataArray((np.arange(sh) * sc) + tr, dims=(ax), attrs={"units": u})
        )

    return DataArray(np.zeros(shape), coords=coords, **kwargs)


def test_ome_ngff_from_arrays():
    axes = ("z", "y", "x")
    units = ("nm", "m", "km")
    translate = (0, -8, 10)
    scale = (1.0, 1.0, 10.0)
    shape = (16,) * 3
    data = create_dataarray(shape, axes, units, scale, translate)
    coarsen_kwargs = {**{dim: 2 for dim in axes}, "boundary": "trim"}
    multi = [data, data.coarsen(**coarsen_kwargs).mean()]
    multi.append(multi[-1].coarsen(**coarsen_kwargs).mean())
    for idx, m in enumerate(multi):
        m.name = f"s{idx}"
    axes, transforms = tuple(zip(*(AxesTransformsFromDataArray(m) for m in multi)))
    group_meta = MultiscaleGroupMetadata.fromDataArrays([multi])
    expected_msg = MultiscaleMetadata(
        name=None,
        version=OmeNgffVersion,
        type=None,
        metadata=None,
        datasets=[
            MultiscaleDataset(path=m.name, coordinateTransformations=transforms[idx])
            for idx, m in enumerate(multi)
        ],
        axes=axes[0],
    )
    expected = MultiscaleGroupMetadata(multiscales=[expected_msg])
    assert group_meta == expected


def test_read_ome_ngff():
    data = np.zeros((16, 16, 16))
    axes = ("z", "y", "x")
    store = zarr.MemoryStore()
    group = zarr.open(store=store, path="foo")
    write_image(data, group, axes=axes)
    group_meta = dict(group.attrs)
    test_meta = MultiscaleGroupMetadata(**group_meta).dict(exclude_none=True)

    assert test_meta == group_meta


def test_write_ome_ngff():
    store_path = tempfile.mkdtemp(suffix=".zarr")
    atexit.register(shutil.rmtree, store_path)

    shape = (16,) * 3
    axes = ("z", "y", "x")
    units = ("m", "nm", "km")
    s0 = create_dataarray(
        shape=shape,
        axes=axes,
        units=units,
        scale=[1, 1, 1],
        translate=[0, 0, 0],
        name="s0",
    )
    s1 = s0.coarsen(dict(zip(axes, (2,) * len(axes)))).mean()
    s1.name = "s1"
    multi = [s0, s1]

    store = zarr.NestedDirectoryStore(store_path)
    group: zarr.Group = zarr.open(store=store, path="foo")
    group_meta = MultiscaleGroupMetadata.fromDataArrays([multi])
    group.attrs.update(**group_meta.dict())
    for m in multi:
        group.create_dataset(name=m.name, data=m.data)

    reader = Reader(parse_url(str(os.path.join(store_path, group.path))))
    assert isinstance(
        tuple(reader())[0].specs[0], ome_zarr.reader.Multiscales
    )  # not clean at all!
