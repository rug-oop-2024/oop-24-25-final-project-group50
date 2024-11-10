from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """NotFoundError class"""
    def __init__(self, path: str) -> None:
        """
        Constructor for NotFoundError class

        Args:
            path: a path which is not found
        Returns:
            None
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Abstract Base Class for storage"""
    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """Storage class for saving artifacts"""
    def __init__(self, base_path: str = "./assets") -> None:
        """
        Constructor for LocalStorage class

        Args:
            base_path: path to be added to every other path
        Returns:
            None
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Saves data in the storage

        Args:
            data: the data to be saved
            key: the location of the storage of the data
        Returns:
            None
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Loads data from the storage

        Args:
            key: the location of the storage of the data
        Returns:
            None
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Deletes data from the storage

        Args
            key: the location of the storage of the data
        Returns:
            None
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        Lists data of the storage

        Args:
            prefix: the location of the storage of the data
        Returns:
            A list with the data in the storage
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path)
                for p in keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Checks if asset path exists

        Args:
            path: the location of the storage of the data
        Returns:
            None
        Raises:
            NotFoundError if the path is not found
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Joins paths together

        Args:
            path: the location of stored data
        Returns:
            the combination of the given path and the base path
        """
        # Ensure paths are OS-agnostic
        return os.path.normpath(os.path.join(self._base_path, path))
