# models/classifiers/simple_emotion_classifier.py
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Iterator, Optional

from models.classifier.abstract_emotion_classifier import AbstractEmotionClassifier

class SimpleEmotionClassifier(AbstractEmotionClassifier):
    """Simple neural network classifier for emotion recognition."""
    
    @classmethod
    def create(cls, 
               embedding_dimension: int, 
               num_classes: int, 
               device: str = "cpu", 
               hidden_dims: list[int] = [128],
               dropout_rate: float = 0.2
            ) -> 'SimpleEmotionClassifier':
        """
        Factory method to create a classifier with standardized parameters.
        
        Args:
            embedding_dimension: Dimension of input embeddings
            num_classes: Number of emotion classes
            device: Device to run model on ('cpu' or 'cuda')
            **kwargs: Additional implementation-specific parameters
                
        Returns:
            An instance of the classifier
        """
        
        # Create and return the instance
        return cls(
            input_dim=embedding_dimension,
            num_emotion_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            device=device
        )
    
    def __init__(
        self, 
        input_dim: int, 
        num_emotion_classes: int,
        hidden_dims: list[int] = [128],
        dropout_rate: float = 0.2,
        device: str = "cpu"
    ):
        """
        Initialize the classifier.
        
        Args:
            input_dim: Dimension of input embeddings
            num_emotion_classes: Number of emotion classes to predict
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            device: Device to run model on ('cpu' or 'cuda')
        """
        self._name = f"SimpleEmotionClassifier-{input_dim}-{hidden_dims}-{num_emotion_classes}"
        self._input_dim = input_dim
        self._num_classes = num_emotion_classes
        self._device = device
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, num_emotion_classes))
        
        # Create the model
        self._model = nn.Sequential(*layers).to(device)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def input_dimension(self) -> int:
        return self._input_dim
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Get model parameters for optimization."""
        return self._model.parameters()
    
    def load_model(self) -> None:
        """Nothing to do as model is initialized in constructor."""
        pass
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self._model(embeddings.to(self._device))
    
    def train(self) -> None:
        """Train the model."""
        self._model.train()
    
    def eval(self) -> None:
        """Evaluate the model."""
        self._model.eval()
    
    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict emotion classes from embeddings."""
        self._model.eval()
        with torch.no_grad():
            logits = self.forward(embeddings)
            _, predictions = torch.max(logits, 1)
        return predictions
    
    def save(self, path: str) -> None:
        """Save the model to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'input_dim': self._input_dim,
            'num_classes': self._num_classes,
            'model_name': self._name,
            'device': self._device
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'SimpleEmotionClassifier':
        """Load a model from a file."""
        checkpoint = torch.load(path)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            num_emotion_classes=checkpoint['num_classes'],
            device=checkpoint.get('device', 'cpu')
        )
        
        model._model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information."""
        # Calculate model size
        size_bytes = 0
        num_parameters = 0
        
        for param in self._model.parameters():
            size_bytes += param.nelement() * param.element_size()
            num_parameters += param.nelement()
        
        size_mb = size_bytes / (1024 * 1024)
        
        return {
            "size_mb": size_mb,
            "parameters": num_parameters
        }