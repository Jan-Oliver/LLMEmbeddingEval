from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any, Iterator, Optional

class AbstractEmotionClassifier(ABC):
    """Abstract base class for all emotion classifier models."""
    
    @classmethod
    @abstractmethod
    def create(cls, 
            embedding_dimension: int, 
            num_classes: int, 
            device: str = "cpu", 
            hidden_dims: list[int] = [128],
            dropout_rate: float = 0.2,
            ) -> 'AbstractEmotionClassifier':
        """
        Factory method to create a classifier with standardized parameters.
        
        Args:
            embedding_dimension: Dimension of input embeddings
            num_classes: Number of emotion classes
            device: Device to run model on ('cpu' or 'cuda')
                
        Returns:
            An instance of the classifier
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier."""
        pass
    
    @property
    @abstractmethod
    def input_dimension(self) -> int:
        """The dimension of input embeddings expected by this model."""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of emotion classes."""
        pass
    
    @abstractmethod
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Get model parameters for optimization.
        
        Returns:
            An iterable of parameters
        """
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            embeddings: Tensor of shape (batch_size, input_dimension)
            
        Returns:
            Tensor of shape (batch_size, num_classes) with logits
        """
        pass
    
    @abstractmethod
    def train(self) -> None:
        """
        Train the model.
        """
        pass
    
    @abstractmethod
    def eval(self) -> None:
        """
        Evaluate the model.
        """
        pass
    
    @abstractmethod
    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict emotion classes from embeddings.
        
        Args:
            embeddings: Tensor of shape (batch_size, input_dimension)
            
        Returns:
            Tensor of shape (batch_size,) with predicted class indices
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'AbstractEmotionClassifier':
        """
        Load a model from a file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        pass
    
    def get_model_size(self) -> Dict[str, Any]:
        """
        Get model size information.
        
        Returns:
            Dictionary with model size details
        """
        # Default implementation that can be overridden by specific models
        return {"size_mb": None, "parameters": None}