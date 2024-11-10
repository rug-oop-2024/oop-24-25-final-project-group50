
from pydantic import BaseModel, Field


class Feature(BaseModel):
    """A description"""
    name: str = Field(default="")
    type: str = Field(default="")

    def __str__(self) -> str:
        """
        Represents the Feature instance as a string when printed.

        Args:
            None
        Returns:
            string representation of the Feature
        """
        return f"Name: {self.name}, Type: {self.type}"
