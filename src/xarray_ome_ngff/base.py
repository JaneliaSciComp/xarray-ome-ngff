from pydantic import BaseModel


class StrictBaseModel(BaseModel):
    """
    A pydantic basemodel that prevents extra fields.
    """

    class config:
        extra = "forbid"
