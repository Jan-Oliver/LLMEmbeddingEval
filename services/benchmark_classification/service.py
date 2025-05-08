import os
import pandas as pd
import argparse
from dotenv import load_dotenv
from typing import Callable, Dict, List, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from data.abstract_dataset_provider import AbstractDatasetProvider
from models.embedding.abstract_embedding_model import AbstractEmbeddingModel
from models.embedding.sentence_transformer_model import SentenceTransformerModel
from models.classifier.abstract_emotion_classifier import AbstractEmotionClassifier
from models.classifier.simple_emotion_classifier import SimpleEmotionClassifier
from services.train.service import TrainService
from services.test.service import TestService
from services.monitoring.service import MonitoringService
from data.twitter_dataset_provider.service import TwitterDatasetProvider
from data.enron_dataset_provider.service import EnronDatasetProvider
from data.synthetic_dataset_provider.service import SyntheticDatasetProvider

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Random state and test size
RANDOM_STATE: int = 42
VAL_SIZE: float = 0.3

# Mapping of emotions to numerical values
EMOTION_MAPPING: dict[str, int] = {
    'Anger': 0,
    'Neutral': 1,
    'Joy': 2,
    'Sadness': 3,
    'Fear': 4,
}

# Define the prefix functions
def prefix_query(text: str) -> str: return "query: " + text
def prefix_passage(text: str) -> str: return "passage: " + text

def prefix_e5_large_instruct(text: str) -> str:
    EMOTION_MAPPING: dict[str, int] = {'Anger': 0, 'Neutral': 1, 'Joy': 2, 'Sadness': 3, 'Fear': 4} # Define this within the function or ensure it's accessible
    return f"Instruct: Classify the emotion expressed in the given Twitter message into one of the {len(EMOTION_MAPPING)} emotions: {', '.join(EMOTION_MAPPING.keys())}.\nQuery: " + text

PREFIX_FUNCTIONS: Dict[str, Optional[Callable[[str], str]]] = {
    "none": None,
    "query": prefix_query,
    "passage": prefix_passage,
    "e5_large_instruct": prefix_e5_large_instruct,
}
def run_classification_benchmark(
        model_name: str,
        normalize_embeddings: bool,
        prefix_function_name: str,
        base_results_folder: str,
        base_dataset_folder: str,
        device: str
    ):
    
    logger.info(f"Starting benchmark for model: {model_name}")
    logger.info(f"Using prefix function: {prefix_function_name} and normalize_embeddings: {normalize_embeddings}")
    
    # Dataset size and path
    DATASET_PATH = Path(base_dataset_folder, "datasets")
    RESULTS_PATH = Path(base_results_folder, f"{model_name}")

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    
    # Init all the dataset providers you want to use
    twitter_dataset_provider = TwitterDatasetProvider(
        output_path=DATASET_PATH, 
        target_emotions=list(EMOTION_MAPPING.keys()),
        openai_api_key=api_key,
        model="gpt-4.1-mini",
        balance_emotions=True,
        max_workers=5,
    )
    
    enron_dataset_provider = EnronDatasetProvider(
        output_path=DATASET_PATH,
        target_emotions=list(EMOTION_MAPPING.keys()),
        target_size=10000,
        openai_api_key=api_key,
        model="gpt-4.1-mini",
        num_classifications=5,
        max_workers=5,
    )
    
    # synthetic_dataset_provider = SyntheticDatasetProvider(
    #     output_path=DATASET_PATH,
    #     target_emotions=list(EMOTION_MAPPING.keys()),
    #     target_size=10000,
    #     openai_api_key=api_key,
    #     model="gpt-4.1-mini",
    #     industries=["finance", "technology", "healthcare", "retail", "manufacturing", "education", "energy", "construction", "hospitality"],
    # )
    
    
    train_dataset: List[AbstractDatasetProvider] = [
        enron_dataset_provider,
        # synthetic_dataset_provider
    ]
    
    test_dataset: List[AbstractDatasetProvider] = [
        twitter_dataset_provider
    ]
    
    # Assert that the datasets in train aren't the same as the ones in test
    assert not any(dataset in test_dataset for dataset in train_dataset)
    
    train_df = pd.DataFrame()
    for dataset in train_dataset:
        # Note: This will create the dataset if it doesn't exist
        df = dataset.get_dataset()
        df.rename(columns={dataset.get_text_column(): 'text'}, inplace=True)
        df.rename(columns={dataset.get_label_column(): 'emotion'}, inplace=True)
        train_df = pd.concat([train_df, df])
        
    test_df = pd.DataFrame()
    for dataset in test_dataset:
        # Note: This will create the dataset if it doesn't exist
        df = dataset.get_dataset()
        df.rename(columns={dataset.get_text_column(): 'text'}, inplace=True)
        df.rename(columns={dataset.get_label_column(): 'emotion'}, inplace=True)
        test_df = pd.concat([test_df, df])
    
    
    # Map emotions to numerical values
    train_df['emotion_label'] = train_df['emotion'].map(EMOTION_MAPPING)
    test_df['emotion_label'] = test_df['emotion'].map(EMOTION_MAPPING)
    
    # Statify by emotion and language
    stratify_labels = train_df['emotion_label'].astype(str) + '_' + train_df['language'].astype(str)

    train_df, val_df = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels
    )
    
    # Initialize monitoring service
    monitoring_service = MonitoringService(
        context_name=RESULTS_PATH.name,
        output_dir=RESULTS_PATH / "monitoring"
    )
    
    
    # Choose an embedding model
    #embedding_model = HuggingFaceModel(model_name=EMBEDDING_MODEL_NAME, monitoring_service=monitoring_service)
    embedding_model = SentenceTransformerModel(
        model_name=model_name, 
        monitoring_service=monitoring_service, 
        normalize_embeddings=normalize_embeddings,
        prefix_function=PREFIX_FUNCTIONS[prefix_function_name],
        device=device
    )
    
    embedding_model.load_model()
    
    # Will be initialized in training service
    classifier = SimpleEmotionClassifier
    
    # Initialize training service with our classifier class
    train_service = TrainService(
        embedding_model=embedding_model,
        classifier_class=classifier,
        num_classes=len(EMOTION_MAPPING),
        model_output_path=RESULTS_PATH
    )
    
    # Train model if it doesn't exist
    train_metrics = train_service.train(
        train_df=train_df,
        val_df=val_df,
        text_column='text',
        label_column='emotion_label',
        emotion_mapping={v: k for k, v in EMOTION_MAPPING.items()},
        batch_size=16,
        epochs=1000,
        learning_rate=1e-4,
        classifier_hidden_dims=[128],
        classifier_dropout_rate=0.2,
        patience = 10, # Number of epochs with no improvement after which training will be stopped
        lr_scheduler_params= {'mode': 'min', 'factor': 0.5, 'patience': 3} # Parameters for ReduceLROnPlateau
    )
    print("Training metrics:", train_metrics)
        
    
    # Now we test the model on the test dataset
    print("Running test ...")
    test_service = TestService(
        embedding_model=embedding_model,
        classifier=train_service.get_best_model(),
        output_path=RESULTS_PATH / "test"
    )

    # Test model and get metrics
    test_metrics = test_service.evaluate(
        df=test_df,
        text_column='text',
        label_column='emotion_label',
        emotion_mapping={v: k for k, v in EMOTION_MAPPING.items()}  # Reverse mapping for confusion matrix labels
    )
    
    print("Test metrics:", test_metrics)
    monitoring_service.save_all_events()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Sentence Embedding Models")
    parser.add_argument("--model-name", required=True, help="Name or path of the Sentence Transformer model")
    parser.add_argument("--normalize-embeddings", action='store_true', help="Use normalize_embeddings=True in model.encode()")
    parser.add_argument("--prefix-function-name", default="none", choices=PREFIX_FUNCTIONS.keys(), help="Name of the prefix function to use")
    parser.add_argument("--base-results-folder", default="results", help="Base folder to save results")
    parser.add_argument("--base-dataset-folder", default="data/datasets", help="Base folder to save datasets")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"], help="Device to run the benchmark on")

    args = parser.parse_args()
    run_classification_benchmark(
        model_name=args.model_name,
        normalize_embeddings=args.normalize_embeddings,
        prefix_function_name=args.prefix_function_name,
        base_results_folder=args.base_results_folder,
        base_dataset_folder=args.base_dataset_folder,
        device=args.device
    )