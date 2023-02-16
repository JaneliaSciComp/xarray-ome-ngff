from xarray import DataArray
import numpy as np

from xarray_ome_ngff.v05.multiscales import (
    create_axes_transforms,
    create_multiscale_metadata,
    create_coords,
)

from pydantic_ome_ngff.v05.axes import Axis
from pydantic_ome_ngff.v05.multiscales import Multiscale, MultiscaleDataset
from pydantic_ome_ngff.v05.coordinateTransformations import (
    VectorScaleTransform,
    VectorTranslationTransform,
)


def create_array(shape, axes, units, types, scale, translate, **kwargs):
    """
    Create a dataarray with a shape and coordinates
    defined by the parameters axes, units, types, scale, translate.
    """
    coords = []
    for ax, u, sh, sc, tr, ty in zip(axes, units, shape, scale, translate, types):
        coords.append(
            DataArray(
                (np.arange(sh) * sc) + tr, dims=(ax), attrs={"units": u, "type": ty}
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
    for idx, m in enumerate(multi):
        m.name = f"s{idx}"
    axes, transforms = tuple(zip(*(create_axes_transforms(m) for m in multi)))
    multiscale_meta = create_multiscale_metadata(multi).dict()
    expected_meta = Multiscale(
        name=None,
        version="0.5-dev",
        type=None,
        metadata=None,
        datasets=[
            MultiscaleDataset(path=m.name, coordinateTransformations=transforms[idx])
            for idx, m in enumerate(multi)
        ],
        axes=axes[0],
        coordinateTransformations=[VectorScaleTransform(scale=[1, 1, 1])],
    ).dict()

    assert multiscale_meta == expected_meta


def test_create_coords():
    shape = (3, 3)
    axes = [
        Axis(name="a", units="meter", type="space"),
        Axis(name="b", units="kilometer", type="space"),
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
            attrs={"units": "meter", "type": "space"},
        )
    )

    assert coords[1].equals(
        DataArray(
            np.array([2.0, 2.5, 3.0]),
            dims=("b",),
            attrs={"units": "kilometer", "type": "space"},
        )
    )
