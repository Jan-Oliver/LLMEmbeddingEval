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
            text_column: Name of column containing text
            label_column: Name of column containing emotion labels
            emotion_mapping: Mapping from label indices to emotion names (for visualization)
            batch_size: Batch size for processing

        Returns:
            Dictionary with evaluation metrics
        """
        # Extract texts and true labels
        texts = df[text_column].tolist()
        true_labels = df[label_column].values.astype(int)  # Ensure integer type

        # Get predictions - This calls self.predict which calculates embeddings
        predicted_labels = self.predict(texts, batch_size=batch_size)
        predicted_labels = predicted_labels.astype(int)  # Ensure integer type

        # Use the new method for metric calculation and visualization
        return self._calculate_and_save_evaluation(true_labels, predicted_labels, emotion_mapping)

    def evaluate_with_embeddings(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        emotion_mapping: Optional[Dict[int, str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model using pre-calculated embeddings and true labels.

        Args:
            embeddings: Pre-calculated embeddings array
            true_labels: Array of true class indices
            emotion_mapping: Mapping from label indices to emotion names (for visualization)

        Returns:
            Dictionary with evaluation metrics
        """
        print("Predicting emotions using provided embeddings...")

        # Convert embeddings to tensor
        inputs = torch.tensor(embeddings, dtype=torch.float32)

        # Get predictions
        predicted_labels = self.classifier.predict(inputs)
        predicted_labels = predicted_labels.cpu().numpy().astype(int) # Ensure integer type

        # Use the new method for metric calculation and visualization
        return self._calculate_and_save_evaluation(true_labels.astype(int), predicted_labels, emotion_mapping)


    def _calculate_and_save_evaluation(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        emotion_mapping: Optional[Dict[int, str]] = None
    ) -> Dict[str, Any]:
        """
        Internal method to calculate metrics, create visualizations, and save results.
        Used by both evaluate and evaluate_with_embeddings.
        """
        # Calculate metrics
        # Handle potential issues if true_labels or predicted_labels contain labels
        # not present in emotion_mapping keys
        all_labels = sorted(list(set(np.concatenate([true_labels, predicted_labels]))))

        if emotion_mapping:
             target_names = [emotion_mapping.get(i, f"Label_{i}") for i in all_labels]
             # Ensure labels argument matches the unique labels present
             labels_for_report = all_labels
             # Create the emotion_mapping dictionary to be used in _create_visualizations,
             # ensuring it includes all labels found in data for robustness.
             viz_emotion_mapping = {i: emotion_mapping.get(i, f"Label_{i}") for i in all_labels}
        else:
             target_names = None
             labels_for_report = None
             viz_emotion_mapping = {i: f"Label_{i}" for i in all_labels} # Create a mapping with default names


        report = classification_report(
            true_labels,
            predicted_labels,
            labels=labels_for_report, # Use detected labels
            target_names=target_names, # Use corresponding names
            output_dict=True,
            zero_division=0 # Handle cases with no true/predicted samples for a class
        )
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=all_labels) # Use all unique labels for matrix

        assert isinstance(report, dict)

        # Create visualizations if output path is provided
        # Pass the viz_emotion_mapping that covers all relevant labels
        if self.output_path:
            self._create_visualizations(conf_matrix, report, viz_emotion_mapping)


        # Save detailed results
        if self.output_path:
             results_data = {
                 'accuracy': report.get('accuracy', 0.0), # Use .get for safety
                 'macro_avg_f1': report.get('macro avg', {}).get('f1-score', 0.0),
                 'weighted_avg_f1': report.get('weighted avg', {}).get('f1-score', 0.0),
                 'confusion_matrix': conf_matrix.tolist(),
                 'classification_report': report
             }
             with open(os.path.join(str(self.output_path), 'evaluation_results.json'), 'w') as f:
                 json.dump(results_data, f, indent=4)


        # Return metrics
        return {
            'accuracy': report.get('accuracy', 0.0),
            'macro_avg_f1': report.get('macro avg', {}).get('f1-score', 0.0),
            'weighted_avg_f1': report.get('weighted avg', {}).get('f1-score', 0.0),
        }


    def _create_visualizations(
        self,
        conf_matrix: np.ndarray,
        report: Dict[str, Any],
        emotion_mapping: Dict[int, str] # Use the mapping that includes ALL labels found in data
    ) -> None:
        """
        Create and save visualizations for model evaluation.

        Args:
            conf_matrix: Confusion matrix from sklearn
            report: Classification report from sklearn
            emotion_mapping: Mapping from label indices to emotion names (includes all labels found in data)
        """
        # Ensure labels for confusion matrix are in the correct order matching the matrix rows/cols
        # Get sorted keys from the mapping passed to this function
        cm_labels = [emotion_mapping[i] for i in sorted(emotion_mapping.keys())]

        # Plot confusion matrix
        plt.figure(figsize=(max(8, len(cm_labels)*1.2), max(6, len(cm_labels)*1.2))) # Adjust size based on number of labels
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=cm_labels, # Use cm_labels here
            yticklabels=cm_labels # Use cm_labels here
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(str(self.output_path), 'confusion_matrix.png'))
        plt.close()

        # Plot precision per class
        # Pass report, metric, title, and emotion_mapping
        self._plot_metric_by_class(report, 'precision', 'Precision per Emotion Class', emotion_mapping)

        # Plot recall per class
        # Pass report, metric, title, and emotion_mapping
        self._plot_metric_by_class(report, 'recall', 'Recall per Emotion Class', emotion_mapping)

        # Plot f1-score per class
        # Pass report, metric, title, and emotion_mapping
        self._plot_metric_by_class(report, 'f1-score', 'F1-score per Emotion Class', emotion_mapping)


    def _plot_metric_by_class(
        self,
        report: Dict[str, Any],
        metric: str,
        title: str,
        emotion_mapping: Dict[int, str] # Use the mapping that includes ALL labels found in data
    ) -> None:
        """
        Create a bar plot for a specific metric by class.

        Args:
            report: Classification report from sklearn
            metric: Metric to plot (e.g., 'precision', 'recall', 'f1-score')
            title: Plot title
            emotion_mapping: Mapping from label indices to emotion names (includes all labels found in data)
        """
        # Get the integer class indices from the emotion_mapping keys and sort them.
        # This list represents the expected order of classes (0, 1, 2, ...)
        sorted_class_indices = sorted(emotion_mapping.keys())

        values = []
        label_names = [] # Generate label names based on the sorted indices

        # Iterate through the sorted integer indices from the emotion mapping
        for i in sorted_class_indices:
             # Get the string name for this integer index using the emotion mapping
             # Use .get for safety, though the map passed from _calculate... should cover all class_indices
             class_name = emotion_mapping.get(i, f"Label_{i}")

             # Get metric value safely using the **class_name** as the key in the report
             # report[class_name] gives the dict for that class name (e.g., {'precision': 0.8, ...})
             class_report_dict = report.get(class_name, {}) # Get the dict for this class name, default to {}
             metric_value = class_report_dict.get(metric, 0.0) # Get the specific metric value from that dict

             values.append(metric_value)
             label_names.append(class_name) # Add the class name to the labels list


        # Check if any class metrics were found based on the mapping
        if not sorted_class_indices:
            print(f"Warning: No class indices found in the emotion mapping. Skipping plot '{title}'.")
            return # Exit the function if no data to plot

        # Check if values list is empty or all zeros (or very close to zero)
        # Use a small epsilon for float comparison
        if not values or max(values) < 1e-6:
             print(f"Info: All values for metric '{metric}' are zero (or effectively zero). Skipping plot '{title}'.")
             return # Skip plotting if all values are zero


        plt.figure(figsize=(max(10, len(label_names)*1.5), 6)) # Adjust size based on number of labels
        bars = plt.bar(label_names, values)
        plt.xlabel('Emotion')
        plt.ylabel(metric.replace('-', ' ').capitalize()) # Format ylabel nicely
        plt.title(title)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha="right") # Rotate labels if many

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
        plt.savefig(os.path.join(str(self.output_path), f'{metric.replace("-", "_")}_per_class.png')) # Save with underscores
        plt.close()
