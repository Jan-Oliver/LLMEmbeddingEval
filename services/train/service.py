# services/train/service.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
# Import the learning rate scheduler module
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Type

from models.embedding.abstract_embedding_model import AbstractEmbeddingModel
from models.classifier.abstract_emotion_classifier import AbstractEmotionClassifier
from services.test.service import TestService

class TrainService:
    """Service for training emotion recognition models using embeddings."""

    def __init__(
        self,
        embedding_model: AbstractEmbeddingModel,
        classifier_class: Type[AbstractEmotionClassifier],
        model_output_path: Path,
        num_classes: int = 6,
        device: str = "cpu"
    ):
        """
        Initialize the training service.

        Args:
            embedding_model: Embedding model to use for text representation
            classifier_class: Class to use for the emotion classifier
            num_classes: Number of emotion classes
            model_output_path: Path to save trained model
            device: Device to train on ('cpu' or 'cuda')
        """
        self.embedding_model = embedding_model
        self.classifier_class = classifier_class
        self.num_classes = num_classes
        self.model_output_path = model_output_path
        self.device = device
        # Ensure output directory exists
        if model_output_path:
            os.makedirs(model_output_path, exist_ok=True)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        text_column: str,
        label_column: str,
        emotion_mapping: Dict[int, str],
        batch_size: int = 16,
        epochs: int = 1000,
        learning_rate: float = 1e-4,
        classifier_hidden_dims: List[int] = [128],
        classifier_dropout_rate: float = 0.2,
        patience: int = 5, # Number of epochs with no improvement after which training will be stopped
        lr_scheduler_params: dict = {'mode': 'min', 'factor': 0.1, 'patience': 2} # Parameters for ReduceLROnPlateau
    ) -> Dict[str, Any]:
        """
        Train an emotion classification model with learning rate scheduling and early stopping.

        Args:
            train_df: DataFrame containing training text and labels
            val_df: DataFrame containing validation text and labels
            text_column: Name of column containing text
            label_column: Name of column containing emotion labels
            emotion_mapping: Mapping from label indices to emotion names
            batch_size: Batch size for training and validation
            epochs: Maximum number of training epochs
            learning_rate: Initial learning rate for optimizer
            device: Device to train on ('cpu' or 'cuda')
            patience: Number of epochs with no improvement in validation loss to wait before early stopping
            lr_scheduler_params: Dictionary of parameters for the ReduceLROnPlateau learning rate scheduler

        Returns:
            Dictionary with training metrics
        """

        print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")

        # Generate embeddings for training set
        print("Generating embeddings for training set...")
        train_texts = train_df[text_column].tolist()
        classes = train_df[label_column].unique()
        train_labels = train_df[label_column].values
        
        # Calculate class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels) # type: ignore
        
        print("Classes: ", classes)
        print("Class weights: ", class_weights)
        # Convert to a PyTorch tensor and move to the appropriate device
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

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
            device=self.device,
            hidden_dims=classifier_hidden_dims,
            dropout_rate=classifier_dropout_rate
        )
        
        # Define loss function and optimizer
        #criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

        # Initialize the learning rate scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_params)

        # Training loop
        best_val_loss = float('inf')
        epochs_no_improve = 0 # Counter for early stopping
        early_stop_triggered = False # Flag for early stopping

        metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [] # Track learning rate
        }

        # Loop for a maximum of 'epochs' or until early stopping
        for epoch in range(epochs):
            # Training phase
            classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

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
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = classifier.forward(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # Step the learning rate scheduler based on validation loss
            scheduler.step(val_loss)

            # Get current learning rate for logging
            current_lr = optimizer.param_groups[0]['lr']

            # Save metrics
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            metrics['learning_rate'].append(current_lr)


            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Current LR: {current_lr:.6f}")

            # Check for improvement and apply early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0 # Reset counter
                # Save best model
                if self.model_output_path:
                    model_path = os.path.join(self.model_output_path, 'model.pt')
                    classifier.save(model_path)

                    # Save model metadata for the best model
                    with open(os.path.join(self.model_output_path, 'model_info.json'), 'w') as f:
                        json.dump({
                            'embedding_model_name': self.embedding_model.name,
                            'embedding_dimension': embedding_dim,
                            'classifier_class': classifier.__class__.__name__,
                            'num_classes': self.num_classes,
                            'epoch': epoch, # Save epoch of the best model
                            'val_loss': val_loss,
                            'val_acc': val_acc,
                            'learning_rate': current_lr # Save LR of the best model epoch
                        }, f, indent=4)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {patience} epochs with no improvement in validation loss.")
                    early_stop_triggered = True
                    break # Exit the training loop

        # Save all training metrics
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

        print("Training complete. Performing validation evaluation with the best model...")
        
        # --- Perform Validation Evaluation using the best model ---
        validation_eval_metrics = {}
        try:
            # Load the best model saved during training
            best_classifier = self.get_best_model()

            # Initialize TestService specifically for this validation evaluation run
            val_eval_output_path = self.model_output_path / "evaluation"
            val_evaluator = TestService(
                    embedding_model=self.embedding_model, # Pass the original embedder (though not used by evaluate_with_embeddings)
                    classifier=best_classifier,
                    output_path=val_eval_output_path # Save results in 'evaluation' subfolder
                )

            # Perform evaluation using the pre-calculated validation embeddings and labels
            validation_eval_metrics = val_evaluator.evaluate_with_embeddings(
                embeddings=val_embeddings,
                true_labels=np.array(val_labels),
                emotion_mapping=emotion_mapping
            )
            
            print("Validation evaluation complete. Results saved to:", val_eval_output_path)

        except FileNotFoundError:
             print("Warning: Best model file not found after training. Skipping validation evaluation.")
        except Exception as e:
             print(f"An error occurred during validation evaluation: {e}")
             validation_eval_metrics['evaluation_error'] = str(e)


        # Prepare the final return dictionary
        final_results = {
            'epochs_ran': epoch + 1, # Number of epochs that actually ran
            'early_stop_triggered': early_stop_triggered,
            'embedding_model': self.embedding_model.name,
            'classifier': classifier.__class__.__name__,
            'training_metrics_summary': { # Summary of last epoch's training metrics
                'last_train_loss': metrics['train_loss'][-1] if metrics['train_loss'] else 'N/A',
                'last_train_acc': metrics['train_acc'][-1] if metrics['train_acc'] else 'N/A',
                'last_val_loss': metrics['val_loss'][-1] if metrics['val_loss'] else 'N/A',
                'last_val_acc': metrics['val_acc'][-1] if metrics['val_acc'] else 'N/A',
            },
            'best_model_metrics_at_save': {}, # Metrics from the epoch where the best model was saved
            'validation_evaluation_results': validation_eval_metrics # Add the evaluation results
        }

        # Load and add best model info if available
        best_model_info_path = os.path.join(self.model_output_path, 'model_info.json')
        if os.path.exists(best_model_info_path):
             with open(best_model_info_path, 'r') as f:
                 best_model_info = json.load(f)
                 final_results['best_model_metrics_at_save'] = {
                    'best_epoch': best_model_info.get('epoch', 'N/A'),
                    'best_val_loss': best_model_info.get('val_loss', 'N/A'),
                    'best_val_acc': best_model_info.get('val_acc', 'N/A'),
                    'best_learning_rate': best_model_info.get('learning_rate', 'N/A')
                 }

        return final_results

    def get_best_model(self) -> AbstractEmotionClassifier:
        """
        Get the best trained model.

        Returns:
            The best trained classifier
        Raises:
            FileNotFoundError: If the best model file does not exist.
        """
        model_path = os.path.join(self.model_output_path, 'model.pt')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Best model file not found at {model_path}. Ensure training was successful and model_output_path was provided.")

        # To correctly load the model, we might need classifier_kwargs used during training.
        # This information is saved in model_info.json.
        model_info_path = os.path.join(self.model_output_path, 'model_info.json')
        return self.classifier_class.load(model_path)