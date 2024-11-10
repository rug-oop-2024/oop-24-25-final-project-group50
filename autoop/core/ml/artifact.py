
from copy import deepcopy
import base64


class Artifact:
    """Stores and contains information about different assets."""

    def __init__(self,
                 name: str = "",
                 data: bytes = None,
                 type: str = "",
                 asset_path: str = "",
                 version: str = "",
                 tags: list = [],
                 metadata: dict = {},
                 ) -> None:
        """
        Constructor for the class Artifact.

        Args:
            name (str): The name of the asset
            version (str): The version of the asset
            asset_path (str): The asset path
            tags (list): A list of tags associated with the asset
            metadata (dict): A dict of internal data
            data (bytes): An encoded version of the asset's data
            type (str): The type of asset that the artifact contains
        Returns:
            None
        """
        self._name = name
        self._asset_path = asset_path
        self._type = type
        self._data = data
        self._version = version
        self._tags = tags
        self._metadata = metadata

    @property
    def name(self) -> str:
        """Get name attribute

        Returns:
            The name
        """
        return self._name

    @property
    def asset_path(self) -> str:
        """Get asset_path attribute

        Returns:
            The asset_path
        """
        return self._asset_path

    @property
    def type(self) -> str:
        """Get type attribute

        Returns:
            The type
        """
        return self._type

    @property
    def data(self) -> bytes:
        """Get data attribute

        Returns:
            The data
        """
        return self._data

    @property
    def version(self) -> str:
        """Get version attribute

        Returns:
            The version
        """
        return self._version

    @property
    def tags(self) -> list:
        """Get tags attribute

        Returns:
            The tags
        """
        return deepcopy(self._tags)

    @property
    def metadata(self) -> dict:
        """Get metadata attribute

        Returns:
            The metadata
        """
        return deepcopy(self._metadata)

    @property
    def id(self) -> str:
        """Get the id

        Returns:
            The id
        """
        id = f"{base64.b64encode(self.name.encode())}_{self.version}"
        id.replace("=", "_")

        return id

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
        """
        Saves the data

        Args:
            bytes: The data to be saved
        Returns:
            The saved data
        """
        self.data = bytes
        return self.data
