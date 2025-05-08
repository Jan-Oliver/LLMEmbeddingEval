from data.enron_dataset_provider.service import EnronDatasetProvider
from pathlib import Path
from dotenv import load_dotenv
import os

target_emotions = ["Anger", "Neutral", "Joy", "Sadness", "Fear"]

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set")


enron_provider = EnronDatasetProvider(
    output_path=Path("data/larger-datasets-enron"),
    target_emotions=target_emotions,
    openai_api_key=openai_api_key,
    target_size=40000,
    num_classifications=5,
    max_workers=5,
    model="gpt-4.1-mini",
)

df = enron_provider.get_dataset()