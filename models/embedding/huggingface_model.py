# models/embeddings/huggingface_model.py
from tqdm import tqdm
from models.embedding.abstract_embedding_model import AbstractEmbeddingModel
import numpy as np
from typing import List, Dict, Any
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

import torch

from services.monitoring.service import MonitoringService

class HuggingFaceModel(AbstractEmbeddingModel):
    """Implementation of AbstractEmbeddingModel using HuggingFace Transformers."""
    
    def __init__(self, model_name: str, monitoring_service: MonitoringService, pooling_strategy: str = "mean"):
        """
        Initialize the model.
        
        Args:
            model_name: Model name or path (e.g., 'bert-base-uncased')
            pooling_strategy: How to convert token embeddings to text embedding 
                              ('mean', 'cls', or 'pooler')
        """
        self._name = model_name
        self._monitoring_service = monitoring_service
        self._pooling_strategy = pooling_strategy
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def embedding_dimension(self) -> int:   
        assert self._model is not None, "Model should be loaded before accessing the embedding dimension"
        return self._model.config.hidden_size
    
    def load_model(self) -> None:
        """Load the model into memory."""
        self._monitoring_service.start_model_load_monitoring()
        self._tokenizer = AutoTokenizer.from_pretrained(self._name)
        self._model = AutoModel.from_pretrained(self._name)
        self._model = self._model.to("cpu")
        # Run a dummy forward pass to ensure the model is loaded
        self._model.eval()
        self._model(**self._tokenizer("Hello, world!", return_tensors="pt"))
        model_load_metrics = self._monitoring_service.stop_model_load_monitoring()
        print(f"Load Metrics: Duration: {model_load_metrics.duration_seconds:.2f}s, RAM Delta: {model_load_metrics.ram_usage_delta_mb:.2f}MB")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts."""        
        all_embeddings = []
        
        self._monitoring_service.start_inference_monitoring(num_texts=len(texts), total_chars=sum(len(text) for text in texts))
        # Process in batches
        with tqdm(total=len(texts), desc=f"Embedding with {self._name}") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize and prepare inputs
                inputs = self._tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                
                # Move to CPU explicitly
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self._model(**inputs)
                
                # Apply pooling strategy
                if self._pooling_strategy == "mean":
                    # Mean pooling across token dimension
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1) / torch.sum(attention_mask, 1)
                elif self._pooling_strategy == "cls":
                    # Use [CLS] token embedding
                    embeddings = outputs.last_hidden_state[:, 0]
                elif self._pooling_strategy == "pooler":
                    # Use model's pooler output if available
                    embeddings = outputs.pooler_output
                else:
                    raise ValueError(f"Unknown pooling strategy: {self._pooling_strategy}")
                
                all_embeddings.append(embeddings.cpu().numpy())
                
                # Update progress bar
                pbar.update(len(batch_texts))
        
        self._monitoring_service.stop_inference_monitoring()
        # Concatenate all batch embeddings
        return np.vstack(all_embeddings)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed_texts([text], batch_size=1)[0]
    
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