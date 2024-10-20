from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    name: str = Field(default="")
    version: str = Field(default="")
    asset_path: str = Field(default="")
    tags: list = Field(default=None)
    metadata: dict = Field(default=None)
    data: bytes = Field(default=None)
    type: str = Field(default="")

    def read(self) -> bytes:
        return self.data
    
    def save(self, bytes: bytes) -> bytes:
        pass
