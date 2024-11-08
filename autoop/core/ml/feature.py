
from pydantic import BaseModel, Field


class Feature(BaseModel):
    """A description"""
    name: str = Field(default="")
    type: str = Field(default="")

    def __str__(self):
        return f"Name: {self.name}, Type: {self.type}"
