from typing import Any, Dict, List, Sequence, Tuple, Union, Optional
import numpy as np
from xarray import DataArray
from pydantic_ome_ngff.v05.axes import Axis
from pydantic_ome_ngff.v05.multiscales import MultiscaleDataset, Multiscale
from pydantic_ome_ngff.v05.coordinateTransformations import (
    VectorScaleTransform,
    VectorTranslationTransform,
    CoordinateTransform,
)
import builtins

JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]


def create_multiscale_metadata(
    arrays: Sequence[DataArray],
    name: Optional[str] = None,
    type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    for arr in arrays:
        if not isinstance(arr, DataArray):
            raise ValueError(
                f"""
                This function requires a list of xarray.DataArrays. 
                Got an element with type {builtins.type(arr)} instead.
                """
            )
    # sort arrays by decreasing shape
    ranks = [a.ndim for a in arrays]
    if len(set(ranks)) > 1:
        raise ValueError(
            f"""
        All arrays must have the same number of dimensions. 
        Found arrays with different numbers of dimensions: {set(ranks)}.
        """
        )
    arrays_sorted = tuple(reversed(sorted(arrays, key=lambda arr: np.prod(arr.shape))))
    base_transforms = [
        VectorScaleTransform(
            scale=[
                1,
            ]
            * ranks[0]
        )
    ]
    axes, transforms = tuple(
        zip(*(create_axes_transforms(array) for array in arrays_sorted))
    )
    paths = [d.name for d in arrays_sorted]
    datasets = list(
        MultiscaleDataset(path=p, coordinateTransformations=t)
        for p, t in zip(paths, transforms)
    )
    return Multiscale(
        version="0.5-dev",
        name=name,
        type=type,
        axes=axes[0],
        datasets=datasets,
        metadata=metadata,
        coordinateTransformations=base_transforms,
    )


def create_axes_transforms(
    array: DataArray,
) -> Tuple[List[Axis], Tuple[VectorScaleTransform, VectorTranslationTransform]]:
    """
    Generate Axes and CoordinateTransformations from an xarray.DataArray.
    """
    translate = []
    scale = []
    axes = []
    for d in array.dims:
        try:
            coord = array[d]
        except KeyError:
            raise ValueError(
                f"""
                Dimension {d} does not have coordinates. 
                All dimensions must have coordinates.
                """
            )

        if len(coord) <= 1:
            raise ValueError(
                f"""
                Cannot infer scale parameter along dimension {d} 
                with length {len(coord)}
                """
            )
        translate.append(float(coord[0]))
        scale.append(abs(float(coord[1]) - float(coord[0])))
        axes.append(
            Axis(
                name=d,
                unit=coord.attrs.get("units", None),
                type=coord.attrs.get("type", None),
            )
        )

    transforms = (
        VectorScaleTransform(scale=scale),
        VectorTranslationTransform(translation=translate),
    )
    return axes, transforms


def create_coords(
    axes: List[Axis], transforms: List[CoordinateTransform], shape: Tuple[int, ...]
) -> Dict[str, DataArray]:
    """
    Given an output shape, convert a sequence of Axis objects
    and a corresponding sequence of transform objects
    into xarray-compatible coordinates.
    """

    if len(axes) != len(shape):
        raise ValueError(
            f"""Length of axes must match length of shape. 
            Got {len(axes)} axes but shape has {len(shape)} elements"""
        )
    result = {}

    for idx, axis in enumerate(axes):
        base_coord = np.arange(shape[idx], dtype="float")
        name = axis.name
        units = axis.units
        # apply transforms in order
        for tx in transforms:
            if type(getattr(tx, "path", None)) == str:
                raise ValueError(
                    f"""
                    Problematic transform: {tx}. 
                    This library is not capable of handling transforms with paths.
                    """
                )

            if tx.type == "translation":
                if len(tx.translation) != len(axes):
                    raise ValueError(
                        f"""
                    Translation parameter has length {len(tx.translation)} 
                    does not match the number of axes {len(axes)}.
                    """
                    )
                base_coord += tx.translation[idx]
            elif tx.type == "scale":
                if len(tx.scale) != len(axes):
                    raise ValueError(
                        f"""
                    Scale parameter has length {len(tx.scale)} 
                    does not match the number of axes {len(axes)}.
                    """
                    )
                base_coord *= tx.scale[idx]
            elif tx.type == "identity":
                pass
            else:
                raise ValueError(
                    f"""
                    Transform type {tx.type} not recognized. 
                    Must be one of scale, translate, or identity
                    """
                )

        result[name] = DataArray(
            base_coord,
            attrs={"units": units},
            dims=(name,),
        )

    return result
