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
import warnings
import pint

JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]

ureg = pint.UnitRegistry()


def create_multiscale(
    arrays: Sequence[DataArray],
    array_paths: Optional[List[str]] = None,
    name: Optional[str] = None,
    type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    normalize_units: bool = True,
    infer_axis_type: bool = True,
):
    """
    Create Multiscale metadata from a collection of xarray.DataArrays

    Parameters
    ----------

    arrays: sequence of DataArray
        The arrays that represent the multiscale collection of images.

    array_paths: sequence of strings, optional
        The path of each array in the group.

    name: string, optional
        The name of the multiscale collection. Used to populate the 'name' field of
        Multiscale metadata.

    type: string, optional
        The type of the multiscale collection. Used to populate the 'type' field of
        Multiscale metadata.

    metadata: dict, optional
        Additional metadata associated with this multiscale collection. Used to populate
        the 'metadata' field of Multiscale metadata.

    normalize_units: bool, defaults to True
        Whether to normalize units to standard names, e.g. 'nm' -> 'nanometer'

    infer_axis_type: bool, defaults to True
        Whether to infer the `type` field of the axis from units, e.g. if units are
        "nanometer" then the type of the axis can safely be assumed to be "space".
        This keyword argument is ignored if `type` is not None in the array coordinate
        metadata. If axis type inference fails, `type` will be set to None.

    Returns
    -------

    An instance of Multiscale metadata.

    """
    for arr in arrays:
        if not isinstance(arr, DataArray):
            raise ValueError(
                f"""
                This function requires a list of xarray.DataArrays. Got an element with 
                type = '{builtins.type(arr)}' instead.
                """
            )
    # sort arrays by decreasing shape
    ranks = [a.ndim for a in arrays]
    if len(set(ranks)) > 1:
        raise ValueError(
            f"""
        All arrays must have the same number of dimensions. Found arrays with different 
        numbers of dimensions: {set(ranks)}.
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
        zip(
            *(
                create_transforms(array, normalize_units=normalize_units)
                for array in arrays_sorted
            )
        )
    )
    if array_paths is None:
        paths = [d.name for d in arrays_sorted]
    else:
        assert len(array_paths) == len(
            arrays
        ), f"""
        Length of array_paths {len(array_paths)} doesn't match {len(arrays)}
        """
        paths = array_paths

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


def create_transforms(
    array: DataArray, normalize_units: bool = True, infer_axis_type=True
) -> Tuple[Tuple[Axis, ...], Tuple[VectorScaleTransform, VectorTranslationTransform]]:
    """
    Generate Axes and CoordinateTransformations from an xarray.DataArray.

    Parameters
    ----------
    array: DataArray
        A DataArray with coordinates for each dimension. Scale and translation
        transform parameters will be inferred from the coordinates for each dimension.
        Note that no effort is made to ensure that the coordinates represent a regular
        grid. Axis types are inferred by querying the attributes of each
        coordinate for the 'type' key. Axis units are inferred by querying the
        attributes of each coordinate for the 'unit' key, and if that key is not present
        then the 'units' key is queried. Axis names are inferred from the dimensions
        of the array.

    normalize_units: bool, defaults to True
        If True, unit strings will be normalized to a canonical representation using the
        `pint` library. For example, the abbreviation "nm" will be normalized to
        "nanometer".

    infer_axis_type: bool, defaults to True
        Whether to infer the axis type from the units. This will have no effect if
        the array has 'type' in its attrs.

    Returns
    -------
        A tuple with two elements. The first value is a tuple of Axis objects with
        length equal to the number of dimensions of the input array. The second value is
        a tuple with two elements which contains a VectorScaleTransform and a
        VectorTranslationTransform, both of which are derived by inspecting the
        coordinates of the input array.
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
                Dimension '{d}' does not have coordinates. All dimensions must have 
                coordinates.
                """
            )
        translate.append(float(coord[0]))
        # impossible to infer a scale coordinate from a coordinate with 1 sample
        if len(coord) > 1:
            scale.append(abs(float(coord[1]) - float(coord[0])))
        else:
            scale.append(1)
        unit = coord.attrs.get("unit", None)
        units = coord.attrs.get("units", None)
        if unit is None and units is not None:
            warnings.warn(
                f"""
            The key 'unit' was unset, but 'units' was found in array attrs, with a value
            of '{units}'. The 'unit' property of the corresponding axis will be set to
            '{units}', but this behavior may change in the future.
            """
            )
            unit = units
        elif units is not None:
            warnings.warn(
                f"""
            Both 'unit' and 'units' were found in array attrs, with values '{unit}' and 
            '{units}', respectively. The value associated with 'unit' ({unit}) will be
            used in the axis metadata.
            """
            )
        if normalize_units:
            unit = ureg.get_name(unit, case_sensitive=True)
        if (type := coord.attrs.get("type", None)) is None and infer_axis_type:
            unit_dimensionality = ureg.get_dimensionality(unit)
            if len(unit_dimensionality) > 1:
                warnings.warn(
                    """
                Failed to infer the type of axis with unit = "{unit}", because it 
                appears that unit "{unit}" is a compound unit, which cannot be mapped
                to a single axis type. "type" will be set to None for this axis.
                """,
                    RuntimeWarning,
                )
            if "[length]" in unit_dimensionality:
                type = "space"
            elif "[time]" in unit_dimensionality:
                type = "time"
            else:
                warnings.warn(
                    """
                Failed to infer the type of axis with unit = "{unit}", because it could 
                not be mapped to either a time or space dimension. "type" will be set to
                None for this axis.
                """,
                    RuntimeWarning,
                )
                type = None

        axes.append(
            Axis(
                name=d,
                unit=unit,
                type=type,
            )
        )

    transforms = (
        VectorScaleTransform(scale=scale),
        VectorTranslationTransform(translation=translate),
    )
    return axes, transforms


def create_coords(
    axes: List[Axis], transforms: List[CoordinateTransform], shape: Tuple[int, ...]
) -> List[DataArray]:
    """
    Given an output shape, convert a sequence of Axis objects and a corresponding
    sequence of coordinateTransform objects into xarray-compatible coordinates.
    """

    if len(axes) != len(shape):
        raise ValueError(
            f"""Length of axes must match length of shape. 
            Got {len(axes)} axes but shape has {len(shape)} elements"""
        )

    result = []

    for idx, axis in enumerate(axes):
        base_coord = np.arange(shape[idx], dtype="float")
        name = axis.name
        unit = axis.unit
        # apply transforms in order
        for tx in transforms:
            if type(getattr(tx, "path", None)) == str:
                raise ValueError(
                    f"""
                    Problematic transform: {tx}. 
                    This library does not handle transforms with paths.
                    """
                )

            if tx.type == "translation":
                if len(tx.translation) != len(axes):
                    raise ValueError(
                        f"""
                    Translation parameter has length {len(tx.translation)}. This does 
                    not match the number of axes {len(axes)}.
                    """
                    )
                base_coord += tx.translation[idx]
            elif tx.type == "scale":
                if len(tx.scale) != len(axes):
                    raise ValueError(
                        f"""
                    Scale parameter has length {len(tx.scale)}. This does not match the 
                    number of axes {len(axes)}.
                    """
                    )
                base_coord *= tx.scale[idx]
            elif tx.type == "identity":
                pass
            else:
                raise ValueError(
                    f"""
                    Transform type {tx.type} not recognized. Must be one of scale, 
                    translate, or identity
                    """
                )

        result.append(
            DataArray(
                base_coord,
                attrs={"unit": unit},
                dims=(name,),
            )
        )

    return result
