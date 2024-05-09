from __future__ import annotations

import os
from typing import Generator, Union

import pint
import zarr
from pydantic import BaseModel
from zarr.errors import ContainsArrayError, GroupNotFoundError, _BaseZarrError

ureg = pint.UnitRegistry()


class CoordinateAttrs(BaseModel):
    """
    A model of the attributes of a DataArray coordinate
    """

    units: str | None


JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]
NGFF_VERSIONS = ("0.4",)


class GroupUnreachableError(GroupNotFoundError):
    _msg = "group could not found because it is outside the Zarr hierarchy"


def get_parent(node: zarr.Group | zarr.Array) -> zarr.Group:
    """
    Get the hierarchically precedent (i.e., parent) node of a Zarr array or group.
    """
    # not all zarr stores have a path attribute, namely memorystore
    if hasattr(node.store, "path"):
        store_path = node.store.path
        if node.path == "":
            new_full_path, new_node_path = os.path.split(os.path.split(store_path)[0])
        else:
            full_path, _ = os.path.split(os.path.join(store_path, node.path))
            new_full_path, new_node_path = os.path.split(full_path)
        return zarr.open_group(
            type(node.store)(new_full_path), path=new_node_path, mode="r"
        )
    else:
        if node.path == "":
            # there is no parent to find, because we are at the root of the hierarchy.
            raise GroupUnreachableError()
        try:
            group_path = os.path.split(node.path)[0]
            return zarr.open_group(store=node.store, path=group_path, mode="r")
        except ContainsArrayError as e:
            raise GroupNotFoundError(group_path) from e


def iter_parents(node: zarr.Group) -> Generator[zarr.Group, None]:
    node_current = node
    while True:
        try:
            parent = get_parent(node_current)
            yield parent
            node_current = parent
        except GroupNotFoundError:
            return
