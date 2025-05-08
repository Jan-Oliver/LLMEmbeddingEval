from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class AbstractDatasetProvider(ABC):
    """
    Abstract base class for dataset providers.
    
    This class defines the common interface that all dataset providers
    (TwitterDatasetService, SyntheticDatasetService, etc.) should implement.
    """
    
    def __init__(self, output_path: Path) -> None:
        """
        Initialize the dataset provider.
        
        Args:
            output_path: Path where the dataset will be saved
        """
        self.output_path = output_path
    
    @abstractmethod
    def dataset_exists(self) -> bool:
        """
        Check if the dataset already exists.
        
        Returns:
            bool: True if the dataset exists, False otherwise
        """
        pass
    
    @abstractmethod
    def ensure_dataset(self) -> None:
        """
        Ensure the dataset exists, creating it if necessary.
        """
        pass
    
    @abstractmethod
    def get_dataset(self) -> pd.DataFrame:
        """
        Get the dataset as a pandas DataFrame.
        If the dataset doesn't exist, this method should create it first.
        
        Returns:
            pd.DataFrame: The dataset
        """
        pass
    
    @abstractmethod
    def get_label_column(self) -> str:
        """
        Get the name of the column that contains the emotion labels.
        
        Returns:
            str: The name of the label column
        """
        pass
    
    @abstractmethod
    def get_text_column(self) -> str:
        """
        Get the name of the column that contains the text data.
        
        Returns:
            str: The name of the text column
        """
        pass