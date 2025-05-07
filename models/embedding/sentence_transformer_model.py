# models/embeddings/sentence_transformer_model.py
from models.embedding.abstract_embedding_model import AbstractEmbeddingModel
import numpy as np
from typing import Callable, List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch

from services.monitoring.service import MonitoringService

class SentenceTransformerModel(AbstractEmbeddingModel):
    """Implementation of AbstractEmbeddingModel using SentenceTransformers."""
    
    def __init__(self, 
                 model_name: str, 
                 monitoring_service: MonitoringService, 
                 normalize_embeddings: bool = False,
                 prefix_function: Optional[Callable[[str], str]] = None):
        """
        Initialize the model.
        
        Args:
            model_name: Model name or path (e.g., 'all-MiniLM-L6-v2')
            language: Language supported by the model
        """
        self._name = model_name
        self._monitoring_service = monitoring_service
        self._normalize_embeddings = normalize_embeddings
        self._prefix_function = prefix_function
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def embedding_dimension(self) -> int:
        dim = self._model.get_sentence_embedding_dimension()
        assert dim is not None, "Model dimension should not be None"
        return dim
    
    def load_model(self) -> None:
        """Load the model into memory."""
        self._monitoring_service.start_model_load_monitoring()
        self._model = SentenceTransformer(self._name, trust_remote_code=True)
        # Run a dummy forward pass to ensure the model is loaded
        self._model.eval()
        self._model.encode("Hello, world!", normalize_embeddings=self._normalize_embeddings)
        model_load_metrics = self._monitoring_service.stop_model_load_monitoring()
        print(f"Load Metrics: Duration: {model_load_metrics.duration_seconds:.2f}s, RAM Delta: {model_load_metrics.ram_usage_delta_mb:.2f}MB")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts."""        
        if self._prefix_function is not None:
            texts = [self._prefix_function(text) for text in texts]
        
        self._monitoring_service.start_inference_monitoring(num_texts=len(texts), total_chars=sum(len(text) for text in texts))
        embeddings = self._model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            normalize_embeddings=self._normalize_embeddings
        )
        self._monitoring_service.stop_inference_monitoring()
        return embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""        
        return self._model.encode(
            text, 
            normalize_embeddings=self._normalize_embeddings
        ).numpy()
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information."""        
        # Calculate model size in MB
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