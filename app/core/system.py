from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    """Class to register artifacts"""
    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """
        Constructor for the class ArtifactRegistery.

        Args:
            database: a database instance
            storage: a storage instance
        Returns:
            None
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Registers data and data keys to storage and datbase.

        Args:
            artifact: an artifact which has to be saved
        Returns:
            None
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists all the artifact in the registery

        Args:
            type: the type of artifact which will be returned
        Returns:
            A list of artifacts of a given type of artifact
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Gets an artifact out of the database, given the id.

        Args:
            artifact_id: the id of an artifact
        Returns:
            The artifact which belongs to the artifact id
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an artifact, given an artifact id.

        Args:
            artifact_id: the id of an artifact
        Returns:
            None
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """Class for managing the AutoMLSystem"""
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Constructor for the class AutoMLSystem.

        Args:
            database: a database instance
            storage: a storage instance
        Returns:
            None
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> 'AutoMLSystem':
        """
        Static method to create an AutoMLSystem instance.

        Args:
            None
        Returns:
            An AutoMLSystem instance
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> 'ArtifactRegistry':
        """
        Gets the ArtifactRegistery of the AutoMLSystem.

        Args:
            None
        Returns:
            The ArtifactRegistery belonging to this AutoMLSystem
        """
        return self._registry
