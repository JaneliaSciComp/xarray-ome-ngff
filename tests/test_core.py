from __future__ import annotations

from typing import TYPE_CHECKING

from xarray_ome_ngff.core import get_parent, iter_parents

if TYPE_CHECKING:
    from typing import Literal
import numpy as np
import pytest
from pydantic_zarr.v2 import ArraySpec, GroupSpec
from zarr import DirectoryStore, MemoryStore, open_array, open_group
from zarr.storage import FSStore
from zarr.errors import GroupNotFoundError

import os
from itertools import accumulate


@pytest.mark.parametrize("node_type", ("array", "group"))
@pytest.mark.parametrize(
    "store", ("directory_store", "memory_store"), indirect=("store",)
)
def test_ascend_once(
    node_type: Literal["array", "group"], store: DirectoryStore | MemoryStore
) -> None:
    tree_flat: dict[str, ArraySpec | GroupSpec] = {
        "/node": GroupSpec(attributes={"target": True})
    }
    if node_type == "array":
        subnode = ArraySpec.from_array(np.arange(10))
    elif node_type == "group":
        subnode = GroupSpec()
    else:
        assert False

    tree_flat["/node/subnode"] = subnode
    hierarchy = GroupSpec.from_flat(tree_flat).to_zarr(store, path="/")

    assert get_parent(hierarchy["node/subnode"]).attrs["target"]


@pytest.mark.parametrize("node_type", ("array", "group"))
@pytest.mark.parametrize(
    "store", ("directory_store", "memory_store"), indirect=("store",)
)
def test_get_parent_root_node(
    node_type: Literal["array", "group"], store: DirectoryStore | MemoryStore
):
    if node_type == "array":
        hierarchy = open_array(store=store, path="", shape=(10, 10))
    elif node_type == "group":
        hierarchy = open_group(store=store, path="")

    with pytest.raises(GroupNotFoundError):
        get_parent(hierarchy)


@pytest.mark.parametrize("start_depth", (1, 2, 3))
@pytest.mark.parametrize(
    "store", ("directory_store", "memory_store", "fsstore_local"), indirect=("store",)
)
def test_iter_parents(start_depth: int, store: MemoryStore | DirectoryStore | FSStore):
    tree_flat = {}
    for key in accumulate(
        map(str, range(1, start_depth)), lambda a, b: os.path.join(a, b), initial="/0"
    ):
        tree_flat[key] = GroupSpec()

    hierarchy = GroupSpec.from_flat(tree_flat).to_zarr(store, path="")
    keys = tuple(tree_flat.keys())
    for key, node in zip(keys[:-1][::-1], iter_parents(hierarchy[keys[-1]])):
        assert node.path.split("/")[-1] == key.split("/")[-1]
