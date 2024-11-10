from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """Class for a dataset instance"""
    def __init__(self, *args, **kwargs) -> None:
        """
        Constructor for the class Dataset.

        Args:
            *args: arguments passed to the artifact constructor
            **kwargs: keyword arguments passed to the artifact constructor
        Returns:
            None
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str, version: str = "1.0.0") -> 'Dataset':
        """
        Static method to turn create a dataset.

        Args:
            data: the data of the dataset
            name: the name of the dataset
            asset_path: the path where the dataset will be saved
            version: the version of the datset
        Returns:
            The Dataset instance created will be returned
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset and turns it to a pandas dataframe.

        Args:
            None
        Returns:
            A pandas dataframe of the dataset
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Reads the pandas dataframe and turns it into bytes.

        Args:
            data: the data given in a pandas dataframe
        Returns:
            the data encoded into bytes
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)

    @staticmethod
    def from_artifact(artifact: Artifact) -> "Dataset":
        """
        Static method which turns an artifact instance to a Dataset instance.

        Args:
            artifact: an Artifact instance
        Returns:
            A Dataset instance transformed from the Artifact instance
        """
        return Dataset(
            name=artifact.name,
            version=artifact.version,
            asset_path=artifact.asset_path,
            tags=artifact.tags,
            metadata=artifact.metadata,
            data=artifact.data
        )
