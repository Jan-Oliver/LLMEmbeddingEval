# models/embeddings/base_embedding_model.py
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union, Dict, Any

class AbstractEmbeddingModel(ABC):
    """Abstract base class for all embedding models."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier."""
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """The dimension of embeddings produced by this model."""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dimension)
        """
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
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