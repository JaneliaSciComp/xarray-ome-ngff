from __future__ import annotations
from pydantic import BaseModel
from typing import Union

import pint

ureg = pint.UnitRegistry()


class CoordinateAttrs(BaseModel):
    """
    A model of the attributes of a DataArray coordinate
    """

    units: str | None


JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]
NGFF_VERSIONS = ("0.4",)
