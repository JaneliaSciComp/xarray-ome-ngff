from pydantic import BaseModel

import pint

ureg = pint.UnitRegistry()


class CoordinateAttrs(BaseModel):
    """
    A model of the attributes of a DataArray coordinate
    """

    units: str | None


JSON = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
NGFF_VERSIONS = ("0.4",)
