
from pathlib import Path
from dotenv import load_dotenv
import os

from data.enron_dataset_provider.service import EnronDatasetProvider

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set")

EMOTION_MAPPING: dict[str, int] = {
    'Anger': 0,
    'Neutral': 1,
    'Joy': 2,
    'Sadness': 3,
    'Fear': 4,
}

enron_dataset_provider = EnronDatasetProvider(
        output_path=Path("data_enron_large/datasets"),
        target_emotions=list(EMOTION_MAPPING.keys()),
        target_size=40000,
        openai_api_key=api_key,
        model="gpt-4.1-mini",
        num_classifications=5,
        max_workers=5,
    )

enron_dataset_provider.get_dataset()