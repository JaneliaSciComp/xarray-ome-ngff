from pydantic import BaseModel
from typing import Union, Dict, List, Optional

import pint

ureg = pint.UnitRegistry()


class CoordinateAttrs(BaseModel):
    """
    A model of the attributes of a DataArray coordinate
    """

    units: Optional[str]


JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]
NGFF_VERSIONS = ("0.4",)
