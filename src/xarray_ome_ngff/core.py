from typing import Union

import pint

ureg = pint.UnitRegistry()

JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]
ngff_versions = ("0.4", "0.5-dev")
