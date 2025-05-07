# services/train/service.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from pathlib import Path
from typing import Dict, Any, Optional, Type

from models.embedding.abstract_embedding_model import AbstractEmbeddingModel
from models.classifier.abstract_emotion_classifier import AbstractEmotionClassifier
from services.monitoring.service import MonitoringService

class TrainService:
    """Service for training emotion recognition models using embeddings."""
    
    def __init__(
        self, 
        embedding_model: AbstractEmbeddingModel,
        classifier_class: Type[AbstractEmotionClassifier],
        model_output_path: Path,
        num_classes: int = 6,
    ):
        """
        Initialize the training service.
        
        Args:
            embedding_model: Embedding model to use for text representation
            classifier_class: Class to use for the emotion classifier
            num_classes: Number of emotion classes
            model_output_path: Path to save trained model
        """
        self.embedding_model = embedding_model
        self.classifier_class = classifier_class
        self.num_classes = num_classes
        self.model_output_path = model_output_path
        
        # Ensure output directory exists
        if model_output_path:
            os.makedirs(model_output_path, exist_ok=True)
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        text_column: str,
        label_column: str,
        batch_size: int = 16,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        device: str = "cpu",
        classifier_kwargs: dict = {}
    ) -> Dict[str, Any]:
        """
        Train an emotion classification model.
        
        Args:
            df: DataFrame containing text and labels
            text_column: Name of column containing email text
            label_column: Name of column containing emotion labels
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cpu' or 'cuda')
            classifier_kwargs: Additional keyword arguments for the classifier constructor
            
        Returns:
            Dictionary with training metrics
        """
        
        print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")
        
        # Generate embeddings for training set
        print("Generating embeddings for training set...")
        train_texts = train_df[text_column].tolist()
        train_labels = train_df[label_column].values
        
        train_embeddings = self.embedding_model.embed_texts(train_texts, batch_size=batch_size)
        
        # Generate embeddings for validation set
        print("Generating embeddings for validation set...")
        val_texts = val_df[text_column].tolist()
        val_labels = val_df[label_column].values
        
        val_embeddings = self.embedding_model.embed_texts(val_texts, batch_size=batch_size)
        
        # Create PyTorch datasets and dataloaders
        train_dataset = TensorDataset(
            torch.tensor(train_embeddings, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long)
        )
        
        val_dataset = TensorDataset(
            torch.tensor(val_embeddings, dtype=torch.float32),
            torch.tensor(val_labels, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        embedding_dim = self.embedding_model.embedding_dimension
        
        # Create classifier with required arguments and any additional kwargs
        classifier = self.classifier_class.create(
            embedding_dimension=embedding_dim,
            num_classes=self.num_classes,
            device=device,
            **classifier_kwargs
        )
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            # Training phase
            classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = classifier.forward(inputs)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            classifier.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = classifier.forward(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # Save metrics
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Save model if path is provided
                if self.model_output_path:
                    model_path = os.path.join(self.model_output_path, 'model.pt')
                    classifier.save(model_path)
                    
                    # Save model metadata
                    with open(os.path.join(self.model_output_path, 'model_info.json'), 'w') as f:
                        json.dump({
                            'embedding_model_name': self.embedding_model.name,
                            'embedding_dimension': embedding_dim,
                            'classifier_class': classifier.__class__.__name__,
                            'num_classes': self.num_classes,
                            **classifier_kwargs,
                            'epoch': epoch,
                            'val_loss': val_loss,
                            'val_acc': val_acc
                        }, f, indent=4)
        
        # Save training metrics
        if self.model_output_path:
            with open(os.path.join(self.model_output_path, 'training_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
                
            # Save embedding model info
            with open(os.path.join(self.model_output_path, 'embedding_info.json'), 'w') as f:
                json.dump({
                    'name': self.embedding_model.name,
                    'type': self.embedding_model.__class__.__name__,
                    'dimension': self.embedding_model.embedding_dimension,
                }, f, indent=4)
        
        return {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epochs': epochs,
            'embedding_model': self.embedding_model.name,
            'classifier': classifier.__class__.__name__
        }
    
    def get_best_model(self) -> AbstractEmotionClassifier:
        """
        Get the best trained model.
        
        Returns:
            The best trained classifier
        """
        model_path = os.path.join(self.model_output_path, 'model.pt')
        return self.classifier_class.load(model_path)