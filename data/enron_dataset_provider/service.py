import csv
import os
import io
import zipfile
import requests
import pandas as pd
from pathlib import Path
from typing import List
from tqdm import tqdm

from ..abstract_dataset_provider import AbstractDatasetProvider
from .email_preprocessor import EnronEmailProcessor
from .email_classifier import EnronEmailClassifier


class EnronDatasetProvider(AbstractDatasetProvider):
    """
    High-level service to check for, preprocess, and provide access to the Enron email dataset.
    Emails are classified with GPT-xx-xxxx model using multiple attempts to get a more accurate result.
    """
    
    def __init__(
        self, 
        output_path: Path, 
        target_emotions: List[str],
        openai_api_key: str,
        target_size: int = 10000,
        num_classifications: int = 5,
        max_workers: int = 5,
        model: str = "gpt-4.1-mini",
    ) -> None:
        """
        Initialize the EnronDatasetProvider.
        
        Args:
            output_path: Path where the processed dataset will be saved
            target_emotions: List of emotion labels to classify
            target_size: Number of emails to sample from the dataset
            num_classifications: Number of times to classify each email
            max_workers: Maximum number of concurrent workers for API calls
            openai_api_key: OpenAI API key
            model: OpenAI model to use for classification
        """
        super().__init__(output_path)
        self.final_output_path = output_path.joinpath("enron", "enron_email_emotions.csv")
        self.raw_data_path = output_path.joinpath("enron", "raw_data.csv")
        self.target_size = target_size
        self.target_emotions = target_emotions
        self.num_classifications = num_classifications
        self.max_workers = max_workers
        self.openai_api_key = openai_api_key
        self.enron_dataset_path = "https://olli-master-thesis.s3.eu-west-1.amazonaws.com/emails.csv.zip"
        self.model = model
        # Create data directory if it doesn't exist
        self.final_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def dataset_exists(self) -> bool:
        """Check if the processed dataset already exists."""
        if not self.final_output_path.exists():
            return False
        
        try:
            with open(self.final_output_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader, None)
                row_count = sum(1 for _ in reader)
                return row_count > 0  # Just check if there's any data
        except Exception as e:
            print(f"Error checking dataset: {e}")
            return False
    
    def _download_and_extract_data(self) -> bool:
        """
        Download the zipped CSV file from S3 and extract it.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.raw_data_path.exists():
            print(f"Raw data already exists at {self.raw_data_path}")
            return True
            
        try:
            print(f"Downloading data from {self.enron_dataset_path}...")
            response = requests.get(str(self.enron_dataset_path), stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Get total file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Setup progress bar
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
            
            # Download the zip file to memory
            zip_content = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    zip_content.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()
            
            # Reset the file pointer to the beginning of the BytesIO object
            zip_content.seek(0)
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            
            # Extract the CSV file from the zip archive
            print("Extracting zip file...")
            with zipfile.ZipFile(zip_content) as zip_ref:
                # List all files in the zip
                file_list = zip_ref.namelist()
                
                # Find the first CSV file
                csv_files = [f for f in file_list if f.endswith('.csv')]
                if not csv_files:
                    print("No CSV files found in the zip archive.")
                    return False
                
                # Extract the first CSV file found
                csv_file = csv_files[0]
                print(f"Extracting {csv_file}...")
                
                # Extract the file
                with zip_ref.open(csv_file) as zf, open(self.raw_data_path, 'wb') as f:
                    f.write(zf.read())
            
            print(f"Successfully downloaded and extracted data to {self.raw_data_path}")
            return True
            
        except Exception as e:
            print(f"Error downloading or extracting data: {e}")
            return False
    
    def _create_dataset(self) -> None:
        """Preprocess the Enron dataset and save it to the output path."""
        # First, download and extract the data
        if not self._download_and_extract_data():
            raise Exception("Failed to download or extract the dataset")
        
        print(f"Loading up to {self.target_size} emails from the Enron dataset...")
        
        # Load and preprocess the emails
        df = self._load_and_preprocess_emails()
        
        # Add emotion labels
        df = self._assign_emotion_labels(df)
        
        # Save to CSV
        df.to_csv(self.final_output_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"Processed dataset saved to {self.final_output_path}")
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"Total samples: {len(df)}")
        if 'final_emotion' in df.columns:
            print("Samples per final emotion category:")
            print(df['final_emotion'].value_counts())
            
            # Also show statistics for each classification attempt
            for i in range(1, self.num_classifications + 1):
                col_name = f'emotion_attempt_{i}'
                if col_name in df.columns:
                    print(f"\nDistribution for classification attempt {i}:")
                    print(df[col_name].value_counts())
    
    def _load_and_preprocess_emails(self) -> pd.DataFrame:
        """Load and preprocess the Enron emails."""
        # Initialize the email processor
        processor = EnronEmailProcessor()
        
        # Read the specified number of rows from the raw data file
        print(f"Reading up to {self.target_size} rows from {self.raw_data_path}...")
        df = pd.read_csv(self.raw_data_path, nrows=self.target_size, skiprows=range(1, 10001))
        
        # Process emails with a progress bar
        print("Preprocessing emails...")
        tqdm.pandas()
        df['text'] = df['message'].progress_apply(processor.preprocess_email)
        
        # Filter out empty messages
        df_filtered = df[df['text'] != ""].reset_index(drop=True)
        
        # Drop duplicate rows
        df_filtered = df_filtered.drop_duplicates(subset='text')
        
        print(f"Loaded {len(df)} emails, {len(df_filtered)} remained after filtering.")
        
        # Select only the necessary columns
        result_df = df_filtered[['text']]
        
        return result_df
    
    def _assign_emotion_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign emotion labels to the emails using the EnronEmailClassifier.
        """
        print("Classifying email emotions with GPT API...")
        
        # Initialize the email classifier
        classifier = EnronEmailClassifier(
            target_emotions=self.target_emotions,
            num_classifications=self.num_classifications,
            max_workers=self.max_workers,
            model=self.model,
            api_key=self.openai_api_key,
            debug=True  # Set to True to see debug output
        )
        
        # Classify the emails
        classified_df = classifier.classify_emails(df, text_column=self.get_text_column())
        
        # Rename columns to match the expected format
        classified_df = classified_df.rename(columns={'final_emotion': 'emotion'})
        
        return classified_df
    
    def ensure_dataset(self) -> None:
        """Ensure the processed dataset exists, creating it if necessary."""
        if self.dataset_exists():
            print(f"Enron dataset already exists at {self.final_output_path}")
        else:
            print("Enron dataset not found. Starting preprocessing...")
            self._create_dataset()
            print("Preprocessing complete.")
    
    def get_dataset(self) -> pd.DataFrame:
        """Get the processed dataset as a DataFrame."""
        self.ensure_dataset()
        return pd.read_csv(self.final_output_path)
    
    def get_label_column(self) -> str:
        """Get the name of the column that contains the emotion labels."""
        return "emotion"
    
    def get_text_column(self) -> str:
        """Get the name of the column that contains the text data."""
        return "text"