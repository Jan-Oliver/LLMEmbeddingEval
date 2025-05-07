# services/test/service.py
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, Any, List, Optional

from models.embedding.abstract_embedding_model import AbstractEmbeddingModel
from models.classifier.abstract_emotion_classifier import AbstractEmotionClassifier

class TestService:
    """Service for testing emotion recognition models using embeddings."""
    
    def __init__(
        self, 
        embedding_model: AbstractEmbeddingModel,
        classifier: AbstractEmotionClassifier,
        output_path: Optional[Path] = None
    ):
        """
        Initialize the test service.
        
        Args:
            embedding_model: Embedding model used for text representation
            classifier: Classifier instance for emotion prediction
            output_path: Path to save evaluation results and visualizations
        """
        self.embedding_model = embedding_model
        self.classifier = classifier
        self.output_path = output_path
        
        # Ensure output directory exists if provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
    
    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Predict emotions for a list of texts.
        
        Args:
            texts: List of text inputs
            batch_size: Batch size for processing
            
        Returns:
            Array of predicted class indices
        """
        print("Generating embeddings...")
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_texts(texts, batch_size=batch_size)
        
        # Convert to tensor
        inputs = torch.tensor(embeddings, dtype=torch.float32)
        
        print("Predicting emotions...")
        # Get predictions
        predictions = self.classifier.predict(inputs)
        
        return predictions.cpu().numpy()
    
    def evaluate(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        emotion_mapping: Optional[Dict[int, str]] = None,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset and generate metrics and visualizations.
        
        Args:
            df: DataFrame with test data
            text_column: Name of column containing email text
            label_column: Name of column containing emotion labels
            emotion_mapping: Mapping from label indices to emotion names (for visualization)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract texts and true labels
        texts = df[text_column].tolist()
        true_labels = df[label_column].values.astype(int)  # Ensure integer type

        # Get predictions
        predicted_labels = self.predict(texts, batch_size=batch_size)
        predicted_labels = predicted_labels.astype(int)  # Ensure integer type
        
        # Calculate metrics
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        
        assert isinstance(report, dict)
        
        # Create visualizations if output path is provided
        if self.output_path and emotion_mapping:
            self._create_visualizations(conf_matrix, report, emotion_mapping)
        
        # Return metrics
        return {
            'accuracy': report['accuracy'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score'],
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report
        }
    
    def _create_visualizations(
        self, 
        conf_matrix: np.ndarray, 
        report: Dict[str, Any], 
        emotion_mapping: Dict[int, str]
    ) -> None:
        """
        Create and save visualizations for model evaluation.
        
        Args:
            conf_matrix: Confusion matrix from sklearn
            report: Classification report from sklearn
            emotion_mapping: Mapping from label indices to emotion names
        """
        labels = [emotion_mapping[i] for i in range(len(emotion_mapping))]
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(str(self.output_path), 'confusion_matrix.png'))
        plt.close()
        
        # Plot precision per class
        self._plot_metric_by_class(labels, report, 'precision', 'Precision per Emotion Class')
        
        # Plot recall per class
        self._plot_metric_by_class(labels, report, 'recall', 'Recall per Emotion Class')
        
        # Save detailed results
        with open(os.path.join(str(self.output_path), 'evaluation_results.json'), 'w') as f:
            json.dump({
                'accuracy': report['accuracy'],
                'macro_avg_f1': report['macro avg']['f1-score'],
                'weighted_avg_f1': report['weighted avg']['f1-score'],
                'classification_report': report
            }, f, indent=4)
    
    def _plot_metric_by_class(
        self, 
        labels: List[str], 
        report: Dict[str, Any], 
        metric: str, 
        title: str
    ) -> None:
        """
        Create a bar plot for a specific metric by class.
        
        Args:
            labels: List of class labels
            report: Classification report from sklearn
            metric: Metric to plot (e.g., 'precision', 'recall')
            title: Plot title
        """
        values = []
        for i in range(len(labels)):
            if str(i) in report:
                values.append(report[str(i)][metric])
            else:
                values.append(0)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, values)
        plt.xlabel('Emotion')
        plt.ylabel(metric.capitalize())
        plt.title(title)
        plt.ylim(0, 1.0)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center'
            )
            
        plt.tight_layout()
        plt.savefig(os.path.join(str(self.output_path), f'{metric}_per_class.png'))
        plt.close()