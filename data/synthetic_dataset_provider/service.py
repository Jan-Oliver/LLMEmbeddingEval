import csv
import os
from pathlib import Path
from typing import List

import pandas as pd

from .dataset_generator import EmailDatasetGenerator
from ..abstract_dataset_provider import AbstractDatasetProvider


class SyntheticDatasetProvider(AbstractDatasetProvider):
    """
    High-level service to check for and generate the synthetic dataset.
    """

    def __init__(self, openai_api_key: str, model: str, target_emotions: List[str], industries: List[str], target_size: int, output_path: Path) -> None:
        self.output_path = output_path.joinpath("synthetic_dataset", "en_train_dataset.csv")
        self.target_size = target_size
        self.target_emotions = target_emotions
        self.industries = industries
        self.generator = EmailDatasetGenerator(
            api_key=openai_api_key, 
            model=model,
            target_size=target_size, 
            output_path=output_path,
            industries=industries,
            emotions=target_emotions
        )

    def dataset_exists(self) -> bool:
        if not self.output_path.exists():
            return False
        
        try:
            with open(self.output_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(
                    f, 
                    quoting=csv.QUOTE_ALL,
                    escapechar='\\'
                )
                row_count = sum(1 for _ in reader)
                return row_count >= self.target_size
        except Exception as e:
            print(f"Error checking dataset: {e}")
            return False

    def _create_dataset(self) -> None:
        self.generator.run()

    def ensure_dataset(self) -> None:
        if self.dataset_exists():
            print(f"Dataset already exists at {self.output_path}")
        else:
            print("Dataset not found. Starting generation...")
            self._create_dataset()
            print("Generation complete.")

    def get_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.output_path)
    
    def get_label_column(self) -> str:
        return "emotion"
    
    def get_text_column(self) -> str:
        return "email_text"
    
    
    
    
