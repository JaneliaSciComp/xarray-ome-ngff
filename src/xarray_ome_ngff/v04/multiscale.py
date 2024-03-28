from __future__ import annotations
from pydantic_ome_ngff.v04.multiscale import Group
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from typing import Any, Dict, List, Sequence, Literal
import numpy as np
from xarray import DataArray
from pydantic_ome_ngff.v04.axis import Axis
from pydantic_ome_ngff.v04.multiscale import Dataset, MultiscaleMetadata
from pydantic_ome_ngff.v04.transform import (
    VectorScale,
    VectorTranslation,
)
import builtins
import warnings
from xarray_ome_ngff.core import ureg


def multiscale_metadata(
    arrays: Sequence[DataArray],
    array_paths: Sequence[str] | None = None,
    name: str | None = None,
    type: str | None = None,
    metadata: Dict[str, Any] | None = None,
    normalize_units: bool = True,
    infer_axis_type: bool = True,
) -> MultiscaleMetadata:
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
            msg = (
                "This function requires a sequence of xarray.DataArrays. Got an element "
                f"with type = '{builtins.type(arr)}' instead."
            )
            raise ValueError(msg)
    # sort arrays by decreasing shape
    ndims = [a.ndim for a in arrays]
    if len(set(ndims)) > 1:
        msg = (
            "All arrays must have the same number of dimensions. Found arrays with "
            f"different numbers of dimensions: {set(ndims)}."
        )
        raise ValueError(msg)
    arrays_sorted = tuple(reversed(sorted(arrays, key=lambda arr: np.prod(arr.shape))))

    base_transforms = [
        VectorScale(
            scale=[
                1,
            ]
            * ndims[0]
        )
    ]
    axes, transforms = tuple(
        zip(
            *(
                coords_to_transforms(
                    tuple(array.coords.values()),
                    normalize_units=normalize_units,
                    infer_axis_type=infer_axis_type,
                )
                for array in arrays_sorted
            )
        )
    )
    if array_paths is None:
        paths = [str(d.name) for d in arrays_sorted]
    else:
        assert len(array_paths) == len(
            arrays
        ), f"Length of array_paths {len(array_paths)} doesn't match {len(arrays)}"
        paths = array_paths

    datasets = list(
        Dataset(path=p, coordinateTransformations=t) for p, t in zip(paths, transforms)
    )
    return MultiscaleMetadata(
        name=name,
        type=type,
        axes=axes[0],
        datasets=datasets,
        metadata=metadata,
        coordinateTransformations=base_transforms,
    )


def coords_to_transforms(
    coords: tuple[DataArray, ...], normalize_units: bool = True, infer_axis_type=True
) -> tuple[tuple[Axis, ...], tuple[VectorScale, VectorTranslation]]:
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
    for coord in coords:
        if ndim := len(coord.dims) != 1:
            msg = (
                "Each coordinate must have one and only one dimension. "
                f"Got a coordinate with {ndim}."
            )
            raise ValueError(msg)
        dim = coord.dims[0]
        translate.append(float(coord[0]))

        # impossible to infer a scale coordinate from a coordinate with 1 sample, so it defaults
        # to 1
        if len(coord) > 1:
            scale.append(abs(float(coord[1]) - float(coord[0])))
        else:
            scale.append(1)
        units = coord.attrs.get("units", None)
        if normalize_units and units is not None:
            units = ureg.get_name(units, case_sensitive=True)
        if (type := coord.attrs.get("type", None)) is None and infer_axis_type:
            unit_dimensionality = ureg.get_dimensionality(units)
            if len(unit_dimensionality) > 1:
                msg = (
                    f'Failed to infer the type of axis with unit = "{units}"',
                    f'because it appears that unit "{units}" is a compound unit, '
                    'which cannot be mapped to a single axis type. "type" will be '
                    'set to "None" for this axis.',
                )
                warnings.warn(
                    RuntimeWarning,
                )
            if "[length]" in unit_dimensionality:
                type = "space"
            elif "[time]" in unit_dimensionality:
                type = "time"
            else:
                msg = (
                    f'Failed to infer the type of axis with unit = "{units}", '
                    "because it could not be mapped to either a time or space "
                    'dimension. "type" will be set to None for this axis.'
                )
                warnings.warn(msg, RuntimeWarning)
                type = None

        axes.append(
            Axis(
                name=dim,
                unit=units,
                type=type,
            )
        )

    transforms = (
        VectorScale(scale=scale),
        VectorTranslation(translation=translate),
    )
    return axes, transforms


def transforms_to_coords(
    axes: List[Axis],
    transforms: Sequence[tuple[VectorScale] | tuple[VectorScale, VectorTranslation]],
    shape: tuple[int, ...],
) -> List[DataArray]:
    """
    Given an output shape, convert a sequence of Axis objects and a corresponding
    sequence of coordinateTransform objects into xarray-compatible coordinates.
    """

    if len(axes) != len(shape):
        msg = (
            "Length of axes must match length of shape. "
            f"Got {len(axes)} axes but shape has {len(shape)} elements"
        )
        raise ValueError(msg)

    result = []

    for idx, axis in enumerate(axes):
        base_coord = np.arange(shape[idx], dtype="float")
        name = axis.name
        unit = axis.unit
        # apply transforms in order
        for tx in transforms:
            if isinstance(getattr(tx, "path", None), str):
                msg = (
                    f"Problematic transform: {tx}. This library cannot handle "
                    "transforms with paths. Resolve this path to a literal scale or "
                    "translation"
                )
                raise ValueError(msg)

            if tx.type == "translation":
                if len(tx.translation) != len(axes):
                    msg = (
                        f"Translation parameter has length {len(tx.translation)}. "
                        f"This does not match the number of axes {len(axes)}."
                    )
                    raise ValueError(msg)
                base_coord += tx.translation[idx]
            elif tx.type == "scale":
                if len(tx.scale) != len(axes):
                    msg = (
                        f"Scale parameter has length {len(tx.scale)}. "
                        f"This does not match the number of axes {len(axes)}."
                    )
                    raise ValueError(msg)
                base_coord *= tx.scale[idx]
            elif tx.type == "identity":
                pass
            else:
                msg = (
                    f"Transform type {tx.type} not recognized. Must be one of scale, "
                    "translation, or identity"
                )
                raise ValueError(msg)

        result.append(
            DataArray(
                base_coord,
                attrs={"units": unit},
                dims=(name,),
            )
        )

    return result


def multiscale_from_arrays(
    arrays: dict[str, DataArray] | Sequence[DataArray],
    paths: Literal["auto"] | Sequence[str],
):
    """
    Create a model of an OME-NGFF multiscale group from a collection of `xarray.DataArray`.
    The dimensions / coordinates of the arrays will be used to infer OME-NGFF axis metadata, as well
    as the OME-NGFF coordinate transformation metadata (i.e., scaling and translation).
    """
    arrays_sorted = tuple(reversed(sorted(arrays, key=lambda arr: np.prod(arr.shape))))
    # pluralses of plurals
    axeses, transformses = tuple(
        zip(*(coords_to_transforms(a.coords) for a in arrays_sorted))
    )

    if len(set(axeses)) != 1:
        raise ValueError(
            f"Got {len(set(axeses))} unique axes from `arrays`",
            "which suggests means that their dimensions and / or coordinates are incompatible.",
        )

    group = Group.from_arrays(
        arrays,
        paths,
        axes=axeses[0],
        scales=[s.scale for s in transformses[0]],
        translations=[t.translation for t in transformses[1]],
    )
    return group


def arrays_from_multiscale(): ...


def normalize_paths(
    paths: Literal["auto"] | Sequence[str],
    arrays: Sequence[DataArray] | dict[str, DataArray],
) -> tuple[tuple[str, DataArray], ...]:
    """
    Normalize the `paths` argument against a sequence of `xarray.DataArray`.
    If `paths` is the string "auto", there are two possibilities:
    - if `arrays` is a `dict`, then the keys of that dict will be validated and returned.
    - if `arrays` is a `Sequence[DataArray]`, then the `name` attribute of each array will be
    validated and returned.
    Otherwise, `paths` is validated and zipped with `arrays` to return a `tuple` of `tuple`, where the
    inner `tuple` are pairs of corresponding `path`, `array` instances.

    Validation entails checking that each element is a string, and that each element appears once.

    Parameters
    ----------
    paths: Literal["auto"] | Sequence[str]
        A specification of the paths of the arrays. Either the string "auto", in which case
        a tuple of strings will be inferred from the `arrays` argument, or a sequence of strings.
    arrays: Sequence[DataArray] | dict[str, DataArray]
        Instances of `xarray.DataArray`, either in a `dict` or a `Sequence`.

    Returns
    -------
    `tuple[tuple[str, xarray.DataArray]]`
        A tuple of array path : array pairs.
    """
    if paths == "auto":
        if isinstance(arrays, Mapping):
            paths_out = tuple(arrays.keys())
        else:
            paths_out = tuple(a.name for a in arrays)
    else:
        if len(paths_out) != len(arrays):
            msg = f"Length of paths ({len(paths)}) does not match length of arrays ({len(arrays)})."
            raise ValueError(msg)
        paths_out = paths

    for path in paths_out:
        if not isinstance(path, str):
            msg = f"Got a non-string path: {path}. Expected a string."
            raise TypeError(msg)
        if paths.count(path) != 1:
            msg = (
                f"An element of paths is repeated: {path}. ",
                "All elements of paths must be unique.",
            )

    return tuple(zip(paths_out, arrays))
