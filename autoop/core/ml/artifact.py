from typing_extensions import Unpack
from pydantic import BaseModel, ConfigDict, Field
import base64


class Artifact(BaseModel):
    """Stores and contains information about different assets.

    Attributes:
        name (str): The name of the asset
        version (str): The version of the asset
        asset_path (str): The asset path
        tags (list): A list of tags associated with the asset
        metadata (dict): A dict of internal data
        data (bytes): An encoded version of the asset's data
        type (str): The type of asset that the artifact contains
    """

    name: str = Field(default="")
    version: str = Field(default="")
    asset_path: str = Field(default="")
    tags: list = Field(default=[])
    metadata: dict = Field(default={})
    data: bytes = Field(default=None)
    type: str = Field(default="")
    id: str = Field(default="")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = f"{base64.b64encode(self.name.encode())}_{self.version}"
        self.id.replace("=", "_")

    def read(self) -> bytes:
        """
        Returns the data

        Args:
            None
        Returns:
            The stored data
        """
        return self.data

    def save(self, bytes: bytes) -> bytes:
        pass
