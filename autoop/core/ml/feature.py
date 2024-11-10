
class Feature:
    """A class for Feature"""

    def __init__(self, name: str = "", type: str = "") -> None:
        """
        Constructor for the class Feature.

        Args:
            name (str): The name of the feature
            type (str): The type of the feature
        Returns:
            None
        """
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        """Get name attribute

        Returns:
            The name
        """
        return self._name

    @property
    def type(self) -> str:
        """Get type attribute

        Returns:
            The type
        """
        return self._type

    def __str__(self) -> str:
        """
        Represents the Feature instance as a string when printed.

        Args:
            None
        Returns:
            string representation of the Feature
        """
        return f"Name: {self.name}, Type: {self.type}"
