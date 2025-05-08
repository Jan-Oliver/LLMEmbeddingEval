import os
import pandas as pd
from dotenv import load_dotenv
from typing import Callable, Dict, List
from pathlib import Path
from sklearn.model_selection import train_test_split
from data.abstract_dataset_provider import AbstractDatasetProvider
from models.embedding.abstract_embedding_model import AbstractEmbeddingModel
from models.embedding.huggingface_model import HuggingFaceModel
from models.embedding.sentence_transformer_model import SentenceTransformerModel
from models.classifier.abstract_emotion_classifier import AbstractEmotionClassifier
from models.classifier.simple_emotion_classifier import SimpleEmotionClassifier
from services.train.service import TrainService
from services.test.service import TestService
from services.monitoring.service import MonitoringService
from data.twitter_dataset_provider.service import TwitterDatasetProvider
from data.enron_dataset_provider.service import EnronDatasetProvider
from data.synthetic_dataset_provider.service import SyntheticDatasetProvider

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


# -----
# Params: 560M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = True
# EMBEDDING_PREFIX_FUNCTION: Callable[[str], str] = lambda x: "query: " + x
# print(EMBEDDING_PREFIX_FUNCTION("Hello, world!"))
# -----
# Params: 560M !TODO: Done
EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_NORMALIZE_EMBEDDINGS: bool = True
EMBEDDING_PREFIX_FUNCTION: Callable[[str], str] = lambda x: f"Instruct: Classify the emotion expressed in the given Twitter message into one of the {len(EMOTION_MAPPING)} emotions: {', '.join(EMOTION_MAPPING.keys())}.\nQuery: " + x
print(EMBEDDING_PREFIX_FUNCTION("Hello, world!"))
# -----
# Params: 278M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-base"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = True
# EMBEDDING_PREFIX_FUNCTION: Callable[[str], str] = lambda x: "query: " + x
# print(EMBEDDING_PREFIX_FUNCTION("Hello, world!"))
# -----
# Params: 118M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = True
# EMBEDDING_PREFIX_FUNCTION: Callable[[str], str] = lambda x: "query: " + x
# print(EMBEDDING_PREFIX_FUNCTION("Hello, world!"))
# -----
# Params: 110M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "google-bert/bert-base-uncased"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = False
# EMBEDDING_PREFIX_FUNCTION = lambda x: x
# -----
# Params: 22.7M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = False
# EMBEDDING_PREFIX_FUNCTION = lambda x: x
# -----
# Params: 33.4M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L12-v2"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = False
# EMBEDDING_PREFIX_FUNCTION = lambda x: x
# -----
# Params: 109M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = False
# EMBEDDING_PREFIX_FUNCTION = lambda x: x
# -----
# Params: 109M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = False
# EMBEDDING_PREFIX_FUNCTION = lambda x: x
# -----
# Params: 82M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-distilroberta-v1"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = False
# EMBEDDING_PREFIX_FUNCTION = lambda x: x
# -----
# Params: 278M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = False
# EMBEDDING_PREFIX_FUNCTION = lambda x: x
# -----
# Params: 278M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "ibm-granite/granite-embedding-278m-multilingual"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = False
# EMBEDDING_PREFIX_FUNCTION = lambda x: x
# -----
# Params: 107M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "ibm-granite/granite-embedding-107m-multilingual"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = False
# EMBEDDING_PREFIX_FUNCTION = lambda x: x
# -----
# Params: 109M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "avsolatorio/GIST-Embedding-v0"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = False
# EMBEDDING_PREFIX_FUNCTION = lambda x: x
# -----
# Params: 33.4M !TODO: Done
# EMBEDDING_MODEL_NAME: str = "avsolatorio/GIST-small-Embedding-v0"
# EMBEDDING_NORMALIZE_EMBEDDINGS: bool = False
# EMBEDDING_PREFIX_FUNCTION = lambda x: x
# -----

LAPTOP: str = "macbook-pro-1"

# Dataset size and path
OUTPUT_PATH = Path(os.path.join(".", "data", "datasets"))
MODEL_OUTPUT_PATH = Path(os.path.join(".", "results", "emotion_classifier", f"{LAPTOP}-{EMBEDDING_MODEL_NAME}"))

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    
    # Init all the dataset providers you want to use
    twitter_dataset_provider = TwitterDatasetProvider(
        output_path=OUTPUT_PATH, 
        target_emotions=list(EMOTION_MAPPING.keys()),
        balance_emotions=True
    )
    
    enron_dataset_provider = EnronDatasetProvider(
        output_path=OUTPUT_PATH,
        target_emotions=list(EMOTION_MAPPING.keys()),
        target_size=10000,
        openai_api_key=api_key,
        model="gpt-4.1-mini",
        num_classifications=5,
        max_workers=5,
    )
    
    # synthetic_dataset_provider = SyntheticDatasetProvider(
    #     output_path=OUTPUT_PATH,
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
    
    train_df, val_df = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df['emotion_label']
    )
    
    # Initialize monitoring service
    monitoring_service = MonitoringService(
        context_name=MODEL_OUTPUT_PATH.name,
        output_dir=MODEL_OUTPUT_PATH / "monitoring"
    )
    
    
    # Choose an embedding model
    #embedding_model = HuggingFaceModel(model_name=EMBEDDING_MODEL_NAME, monitoring_service=monitoring_service)
    embedding_model = SentenceTransformerModel(
        model_name=EMBEDDING_MODEL_NAME, 
        monitoring_service=monitoring_service, 
        normalize_embeddings=EMBEDDING_NORMALIZE_EMBEDDINGS,
        prefix_function=EMBEDDING_PREFIX_FUNCTION
    )
    
    embedding_model.load_model()
    
    # Will be initialized in training service
    classifier = SimpleEmotionClassifier
    
    # Initialize training service with our classifier class
    train_service = TrainService(
        embedding_model=embedding_model,
        classifier_class=classifier,
        num_classes=len(EMOTION_MAPPING),
        model_output_path=MODEL_OUTPUT_PATH
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
        output_path=MODEL_OUTPUT_PATH / "test"
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
    main()