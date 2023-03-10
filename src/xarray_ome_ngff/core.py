from typing import Union, Dict, List

import pint

ureg = pint.UnitRegistry()

JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]
ngff_versions = ("0.4", "0.5-dev")
