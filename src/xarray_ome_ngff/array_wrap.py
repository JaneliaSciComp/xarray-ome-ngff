from __future__ import annotations
from typing import TYPE_CHECKING, cast, runtime_checkable, Protocol, TypedDict

if TYPE_CHECKING:
    from typing import Any, Literal
    from typing_extensions import Self

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.util import find_spec
import numpy as np
import zarr


@runtime_checkable
class Arrayish(Protocol):
    dtype: np.dtype
    shape: tuple[int, ...]

    def __getitem__(self, *args) -> Self: ...


class ArrayWrapperSpec(TypedDict):
    name: Literal["dask"]
    config: dict[str, Any]


class DaskArrayWrapperConfig(TypedDict):
    """
    A model of the attributes of `DaskArrayWrapper`.
    """

    chunks: str | int | tuple[int, ...] | tuple[tuple[int, ...], ...]
    meta: Any = None
    inline_array: bool


class ZarrArrayWrapperSpec(ArrayWrapperSpec):
    name: Literal["zarr_array"]
    config: dict[str, Any] = {}


class DaskArrayWrapperSpec(ArrayWrapperSpec):
    name: Literal["dask_array"]
    config: DaskArrayWrapperConfig


class BaseArrayWrapper(ABC):
    @abstractmethod
    def wrap(self, data: zarr.Array) -> Arrayish: ...


@dataclass
class ZarrArrayWrapper(BaseArrayWrapper):
    """
    An array wrapper that passes a `zarr.Array` instances through unchanged.
    """

    def wrap(self, data: zarr.Array) -> Arrayish:
        return data


@dataclass
class DaskArrayWrapper(BaseArrayWrapper):
    """
    An array wrapper that wraps a `zarr.Array` in a dask array using `dask.array.from_array`.
    The attributes of this class are a subset of the keyword arguments to `dask.array.from_array`;
    specifically, those keyword arguments that make sense when the input to `from_array` is a
    `zarr.Array`.

    Attributes
    ----------
    chunks: str | int | tuple[int, ...]  tuple[tuple[int, ...], ...] = "auto"
        The chunks for the Dask array. See `dask.array.from_array` for details.
    meta: Any = `None`
        The array type of each chunk of the Dask array. See `dask.array.from_array` for details.
    inline_array: bool = True
        Whether slices of this array should be inlined into the Dask task graph.
        See `dask.array.from_array` for details.
    """

    chunks: str | int | tuple[int, ...] | tuple[tuple[int, ...], ...] = "auto"
    meta: Any = None
    inline_array: bool = True

    def __post_init__(self) -> None:
        """
        Handle the lack of `dask`.
        """
        if find_spec("dask") is None:
            msg = (
                "Failed to import `dask` successfully. "
                "The `dask` library is required to use `DaskArrayWrapper`."
                "Install `dask` into your python environment, e.g. via "
                "`pip install dask`, to resolve this issue."
            )
            ImportError(msg)

    def wrap(self, data: zarr.Array):
        """
        Wrap the input in a dask array.
        """
        import dask.array as da  # noqa

        return da.from_array(
            data, chunks=self.chunks, inline_array=self.inline_array, meta=self.meta
        )


def resolve_wrapper(spec: ArrayWrapperSpec) -> BaseArrayWrapper:
    """
    Convert an `ArrayWrapperSpec` into the corresponding `BaseArrayWrapper` subclass.
    """
    if spec["name"] == "dask_array":
        spec = cast(DaskArrayWrapperConfig, spec)
        return DaskArrayWrapper(**spec["config"])
    elif spec["name"] == "zarr_array":
        spec = cast(ZarrArrayWrapperSpec, spec)
        return ZarrArrayWrapper(**spec["config"])
    else:
        raise ValueError(f"Spec {spec} is not recognized.")


def parse_wrapper(data: ArrayWrapperSpec | BaseArrayWrapper):
    """
    Parse the input into a `BaseArrayWrapper` subclass.

    If the input is already `BaseArrayWrapper`, it is returned as-is.
    Otherwise, the input is presumed to be `ArrayWrapperSpec` and is passed to `resolve_wrapper`.
    """
    if isinstance(data, BaseArrayWrapper):
        return data
    return resolve_wrapper(data)
