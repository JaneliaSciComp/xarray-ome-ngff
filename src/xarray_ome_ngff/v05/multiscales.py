from pydantic import BaseModel
from typing import Any, Dict, List, Sequence, Tuple, Union, Literal, Optional
import numpy as np
from xarray import DataArray
from xarray_ome_ngff.base import StrictBaseModel

JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]

OmeNgffVersion = "0.5-dev"

SpaceUnit = Literal[
    "angstrom",
    "attometer",
    "centimeter",
    "decimeter",
    "exameter",
    "femtometer",
    "foot",
    "gigameter",
    "hectometer",
    "inch",
    "kilometer",
    "megameter",
    "meter",
    "micrometer",
    "mile",
    "millimeter",
    "nanometer",
    "parsec",
    "petameter",
    "picometer",
    "terameter",
    "yard",
    "yoctometer",
    "yottameter",
    "zeptometer",
    "zettameter",
]

TimeUnit = Literal[
    "attosecond",
    "centisecond",
    "day",
    "decisecond",
    "exasecond",
    "femtosecond",
    "gigasecond",
    "hectosecond",
    "hour",
    "kilosecond",
    "megasecond",
    "microsecond",
    "millisecond",
    "minute",
    "nanosecond",
    "petasecond",
    "picosecond",
    "second",
    "terasecond",
    "yoctosecond",
    "yottasecond",
    "zeptosecond",
    "zettasecond",
]

AxisType = Literal[
    "space", "time", "channel"
]  # axis types should probably be "dimensional" vs "categorical" instead


class PathTransform(
    StrictBaseModel
):  # the existence of this type is a massive sinkhole in the spec
    type: Union[Literal["scale"], Literal["translation"]]
    path: str


class VectorTranslationTransform(StrictBaseModel):
    type: Literal["translation"] = "translation"
    translation: List[float]  # redundant field name -- we already know it's translation


class VectorScaleTransform(StrictBaseModel):
    type: Literal["scale"] = "scale"
    scale: List[float]  # redundant field name -- we already know it's scale


ScaleTransform = Union[VectorScaleTransform, PathTransform]
TranslationTransform = Union[VectorTranslationTransform, PathTransform]
CoordinateTransform = List[Union[ScaleTransform, TranslationTransform]]


class Axis(StrictBaseModel):
    name: str
    type: Optional[Union[AxisType, str]]  # unit defines type, so this is not needed
    unit: Optional[Union[TimeUnit, SpaceUnit, str]]


class MultiscaleDataset(BaseModel):
    path: str
    coordinateTransformations: Union[
        List[ScaleTransform], List[Union[ScaleTransform, TranslationTransform]]
    ]


class MultiscaleMetadata(BaseModel):
    version: Optional[str] = OmeNgffVersion  # why is this optional?
    name: Optional[str]  # why is this nullable instead of reserving the empty string
    type: Optional[
        str
    ]  # not clear what this field is for, given the existence of .metadata
    metadata: Optional[
        Dict[str, Any]
    ] = None  # should default to empty dict instead of None
    axes: List[
        Axis
    ]  # should not exist at top level and instead live in dataset metadata or in .datasets
    datasets: List[MultiscaleDataset]
    coordinateTransformations: Optional[
        List[Union[ScaleTransform, TranslationTransform]]
    ]  # should not live here, and if it is here, it should default to an empty list instead of being nullable

    @classmethod
    def fromDataArrays(
        cls,
        arrays: Sequence[DataArray],
        name: Optional[str] = None,
        type: Optional[str] = None,
        version: Optional[str] = OmeNgffVersion,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        for arr in arrays:
            if not isinstance(arr, DataArray):
                raise ValueError(
                    f"This function requires a list of xarray.DataArrays. Got an element with type {type(arr)} instead."
                )
        # sort arrays by decreasing shape
        arrays_sorted = tuple(
            reversed(sorted(arrays, key=lambda arr: np.prod(arr.shape)))
        )
        axes, transforms = tuple(
            zip(*(AxesTransformsFromDataArray(array) for array in arrays_sorted))
        )
        paths = [d.name for d in arrays_sorted]
        datasets = list(
            MultiscaleDataset(path=p, coordinateTransformations=t)
            for p, t in zip(paths, transforms)
        )
        return cls(
            version=version,
            name=name,
            type=type,
            axes=axes[0],
            datasets=datasets,
            metadata=metadata,
        )


class MultiscaleGroupMetadata(BaseModel):
    multiscales: List[MultiscaleMetadata]

    @classmethod
    def fromDataArrays(cls, arrays: Sequence[Sequence[DataArray]]):
        """
        Generate a list of MultiscaleMetadata from a sequence of sequences of xarray.DataArray.
        """
        return cls(
            multiscales=[MultiscaleMetadata.fromDataArrays(arrs) for arrs in arrays]
        )


class ArrayMetadata(BaseModel):
    axes: List[Axis]
    coordinateTransformations: List[Union[ScaleTransform, TranslationTransform]]

    @classmethod
    def fromDataArray(cls, array: DataArray) -> "ArrayMetadata":
        """
        Generate an instance of ArrayMetadata from a DataArray.

        Parameters
        ----------

        array: DataArray

        Returns
        -------

        ArrayMetadata

        """
        axes, transforms = AxesTransformsFromDataArray(array)
        return cls(axes=axes, coordinateTransformations=transforms)


def AxesTransformsFromDataArray(
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
                f"Dimension {d} does not have coordinates. All dimensions must have coordinates."
            )

        if len(coord) <= 1:
            raise ValueError(
                f"Cannot infer scale parameter along dimension {d} with length {len(coord)}"
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


def AxesTransformsToCoords(
    axes: List[Axis], transforms: List[CoordinateTransform], shape: Tuple[int]
) -> Dict[str, DataArray]:
    """
    Given an output shape, convert a sequence of Axis objects and a corresponding sequence of transform objects into xarray-compatible
    coordinates.
    """

    if len(axes) != len(transforms):
        raise ValueError(
            f"Length of axes must match length of transforms. Got {len(axes)} axes but {len(transforms)} transforms."
        )
    if len(axes) != len(shape):
        raise ValueError(
            f"Length of axes must match length of shape. Got {len(axes)} axes but shape has {len(shape)} elements"
        )
    result = {}

    for idx, axis in enumerate(axes):
        base_coord = np.arange(shape[idx])
        name = axis.name
        unit = axis.unit
        # apply transforms in order
        for tx in transforms:
            if type(getattr(tx, "path")) == str:
                raise ValueError(
                    f"Problematic transform: {tx}. This library is not capable of handling transforms with paths."
                )

            if tx.type == "translate":
                base_coord += tx.translate[idx]
            elif tx.type == "scale":
                base_coord *= tx.scale[idx]
            elif tx.type == "identity":
                pass
            else:
                raise ValueError(
                    f"Transform type {tx.type} not recognized. Must be one of scale, translate, or identity"
                )

        result[name] = DataArray(
            base_coord,
            attrs={"units": unit},
            dims=(name,),
        )

    return result
