import csv
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..abstract_dataset_provider import AbstractDatasetProvider

class TwitterDatasetProvider(AbstractDatasetProvider):

    """
    High-level service to check for, download, and fuse test datasets.
    """
    
    # URLs for the datasets
    CROWDFLOWER_URL = "https://raw.githubusercontent.com/tlkh/text-emotion-classification/master/dataset/original/text_emotion.csv"
    CBET_URL = "https://webdocs.cs.ualberta.ca/~zaiane/CBET/CBET.csv"
    
    
    def __init__(self, output_path: Path, target_emotions: List[str], balance_emotions: bool = True) -> None:
        """
        Initialize the TestDatasetService.
        
        Args:
            output_path: Path where the final fused dataset will be saved
            balance_emotions: Whether to balance the dataset by emotion category
        """
        self.output_path = output_path.joinpath("twitter_dataset")
        self.crowdflower_path = self.output_path / "crowdflower.csv"
        self.cbet_path = self.output_path / "cbet.csv"
        self.balance_emotions = balance_emotions
        self.target_emotions = target_emotions
        
        # Create data directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Final fused dataset path
        #self.final_output_path = self.output_path / "test_dataset.csv"
        self.final_output_path = self.output_path / "enron_email_emotions.csv"
    
    def dataset_exists(self) -> bool:
        """Check if the fused dataset already exists."""
        if not self.final_output_path.exists():
            return False
        
        try:
            with open(self.final_output_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader, None)
                row_count = sum(1 for _ in reader)
                return row_count > 0
        except Exception as e:
            print(f"Error checking dataset: {e}")
            return False
    
    def _download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a file from a given URL.
        
        Args:
            url: URL to download from
            output_path: Path to save the downloaded file
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"Downloaded {url} to {output_path}")
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def ensure_datasets(self) -> Tuple[bool, bool]:
        """
        Ensure both datasets are available, downloading them if needed.
        
        Returns:
            Tuple[bool, bool]: Tuple indicating if CrowdFlower and CBET datasets are available
        """
        crowdflower_exists = self.crowdflower_path.exists()
        cbet_exists = self.cbet_path.exists()
        
        if not crowdflower_exists:
            print(f"CrowdFlower dataset not found. Downloading from {self.CROWDFLOWER_URL}...")
            crowdflower_exists = self._download_file(self.CROWDFLOWER_URL, self.crowdflower_path)
        else:
            print(f"CrowdFlower dataset found at {self.crowdflower_path}")
        
        if not cbet_exists:
            print(f"CBET dataset not found. Downloading from {self.CBET_URL}...")
            cbet_exists = self._download_file(self.CBET_URL, self.cbet_path)
        else:
            print(f"CBET dataset found at {self.cbet_path}")
        
        return crowdflower_exists, cbet_exists
    
    def _process_crowdflower(self) -> pd.DataFrame:
        """
        Process the CrowdFlower dataset.
        
        Returns:
            pd.DataFrame: Processed CrowdFlower data with mapped target emotions
        """
        
        # Emotion mappings from CrowdFlower
        CROWDFLOWER_MAPPINGS = {
            "Anger": ["anger", "hate"],
            "Neutral": ["neutral"],
            "Joy": ["happiness", "fun", "enthusiasm", "love"],
            "Sadness": ["sadness"],
            "Fear": ["worry"],
            # Not mapping: surprise, relief, empty, boredom
        }
        
        # Make sure the mapping matches the target emotions
        if not set(CROWDFLOWER_MAPPINGS.keys()).issubset(set(self.target_emotions)):
            raise ValueError(f"Target emotions {set(self.target_emotions)} do not match CrowdFlower emotions {set(CROWDFLOWER_MAPPINGS.keys())}")
        
        try:
            df = pd.read_csv(self.crowdflower_path)
            
            # Print column names for debugging
            print(f"CrowdFlower columns: {df.columns.tolist()}")
            
            # Confirm we have the expected columns
            if 'content' not in df.columns or 'sentiment' not in df.columns:
                print(f"Warning: Expected columns not found in CrowdFlower dataset")
                print(f"Available columns: {df.columns.tolist()}")
                
            # Map columns to our standard format
            df = df.rename(columns={
                'content': 'text',
                'sentiment': 'source_emotion'
            })
            
            # Only keep needed columns
            df = df[['text', 'source_emotion']]
            
            # Add source column
            df['source'] = 'crowdflower'
            
            # Create reverse mapping from CrowdFlower emotions to target emotions
            reverse_mapping = {}
            for target, sources in CROWDFLOWER_MAPPINGS.items():
                for source in sources:
                    reverse_mapping[source.lower()] = target
            
            # Map to target emotions
            df['emotion'] = df['source_emotion'].str.lower().map(reverse_mapping)
            
            # Drop rows with emotions we don't want to map
            df = df.dropna(subset=['emotion'])
            
            # Keep only necessary columns
            df = df[['text', 'emotion', 'source']]
            
            return df
        except Exception as e:
            print(f"Error processing CrowdFlower dataset: {e}")
            return pd.DataFrame()
    
    def _process_cbet(self) -> pd.DataFrame:
        """
        Process the CBET dataset.
        
        Returns:
            pd.DataFrame: Processed CBET data with mapped target emotions
        """
        try:
            df = pd.read_csv(self.cbet_path)
            
            # Print column names for debugging
            print(f"CBET columns: {df.columns.tolist()}")
            
            # Ensure 'text' column exists (it might be called 'sentence' or something else)
            if 'text' not in df.columns:
                if 'sentence' in df.columns:
                    df = df.rename(columns={'sentence': 'text'})
                # If there's no text column but there's an 'id' column with text-like content
                elif 's_' in df['id'].iloc[0] and len(df.columns) > 8:
                    print("Warning: No text/sentence column found. Using placeholder.")
                    df['text'] = "Text not available"
                else:
                    raise ValueError("Could not identify text column in CBET dataset")
            
            # The CBET dataset has intensity scores for each emotion in separate columns
            # We need to determine the dominant emotion for each text
            
            # Mapping from CBET emotion columns to our target emotions
            emotion_col_mapping = {
                'anger': 'Anger',
                'joy': 'Joy',
                'sadness': 'Sadness',
                'fear': 'Fear',
                'love': 'Joy',
                'thankfulness': 'Joy'
            }
            
            # Make sure the mapped emotions are in the target emotions
            mapped_emotions = set(emotion_col_mapping.values())
            if not mapped_emotions.issubset(set(self.target_emotions)):
                raise ValueError(f"Mapped emotions {mapped_emotions} are not all in target emotions {set(self.target_emotions)}")
            
            # Check if we have the expected emotion columns
            found_emotions = [col for col in emotion_col_mapping.keys() if col in df.columns]
            if not found_emotions:
                print("Warning: No expected emotion columns found in CBET dataset")
                print(f"Looking for any of: {list(emotion_col_mapping.keys())}")
                print(f"Available columns: {df.columns.tolist()}")
            
            # Create a new DataFrame to store the results
            result_rows = []
            
            # Process each row
            for _, row in df.iterrows():
                text = row['text']
                
                # Get the emotion scores for the emotions we care about
                emotion_scores = {}
                for cbet_emotion, target_emotion in emotion_col_mapping.items():
                    if cbet_emotion in df.columns:
                        # If multiple CBET emotions map to the same target emotion,
                        # take the maximum score
                        current_score = emotion_scores.get(target_emotion, 0.0)
                        new_score = row[cbet_emotion]
                        emotion_scores[target_emotion] = max(current_score, new_score)
                
                # Determine if the text is neutral (all emotion scores are 0)
                all_zero = all(score == 0.0 for score in emotion_scores.values())
                if all_zero:
                    emotion_scores['Neutral'] = 1.0
                
                # Find the dominant emotion (highest score)
                dominant_emotion = None
                max_score = 0.0
                
                for emotion, score in emotion_scores.items():
                    if score > max_score:
                        max_score = score
                        dominant_emotion = emotion
                
                # Only add rows with a valid dominant emotion and non-zero score
                if dominant_emotion and max_score > 0.0:
                    result_rows.append({
                        'text': text,
                        'emotion': dominant_emotion,
                        'source': 'cbet',
                        'score': max_score
                    })
            
            # Create DataFrame from processed rows
            result_df = pd.DataFrame(result_rows)
            
            # Keep only necessary columns
            if not result_df.empty:
                result_df = result_df[['text', 'emotion', 'source']]
                print(f"Successfully processed {len(result_df)} rows from CBET dataset")
            else:
                print("Warning: No valid rows extracted from CBET dataset")
            
            return result_df
        except Exception as e:
            print(f"Error processing CBET dataset: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _fuse_datasets(self) -> pd.DataFrame:
        """
        Fuse the CrowdFlower and CBET datasets.
        
        Returns:
            pd.DataFrame: Fused dataset with consistent target emotions
        """
        # Process individual datasets - they already map to target emotions
        crowdflower_df = self._process_crowdflower()
        cbet_df = self._process_cbet()
        
        print(f"Processed CrowdFlower dataset: {len(crowdflower_df)} rows")
        print(f"Processed CBET dataset: {len(cbet_df)} rows")
        
        if not crowdflower_df.empty:
            print(f"CrowdFlower emotion distribution:\n{crowdflower_df['emotion'].value_counts()}")
        
        if not cbet_df.empty:
            print(f"CBET emotion distribution:\n{cbet_df['emotion'].value_counts()}")
        
        # Combine datasets
        combined_df = pd.concat([crowdflower_df], ignore_index=True)
        
        # Ensure we have only our target emotions
        valid_emotions = set(self.target_emotions)
        combined_df = combined_df[combined_df['emotion'].isin(valid_emotions)]
        
        return combined_df
    
    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset to have an equal number of samples per emotion.
        
        Args:
            df: DataFrame with 'emotion' column
            
        Returns:
            pd.DataFrame: Balanced DataFrame
        """
        if not self.balance_emotions:
            return df
            
        # Count samples per emotion
        emotion_counts = df['emotion'].value_counts()
        min_count = emotion_counts.min()
        
        # Sample equally from each emotion
        balanced_dfs = []
        for emotion in self.target_emotions:
            if emotion in emotion_counts:
                emotion_df = df[df['emotion'] == emotion]
                # If we have more samples than min_count, sample randomly
                if len(emotion_df) > min_count:
                    emotion_df = emotion_df.sample(min_count, random_state=42)
                balanced_dfs.append(emotion_df)
        
        # Combine balanced dataframes
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        return balanced_df
    
    def _create_dataset(self) -> None:
        """Create the fused dataset and save it to the output path."""
        fused_df = self._fuse_datasets()
        
        # Balance the dataset if required
        if self.balance_emotions:
            before_balance = len(fused_df)
            fused_df = self._balance_dataset(fused_df)
            after_balance = len(fused_df)
            print(f"Dataset balanced: {before_balance} samples -> {after_balance} samples")
        
        # Shuffle the dataset
        fused_df = fused_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to CSV
        fused_df.to_csv(self.final_output_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"Fused dataset saved to {self.final_output_path}")
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"Total samples: {len(fused_df)}")
        print("Samples per emotion category:")
        print(fused_df['emotion'].value_counts())
        print("\nSamples per source:")
        print(fused_df['source'].value_counts())
    
    def ensure_dataset(self) -> None:
        """Ensure the fused dataset exists, creating it if necessary."""
        if self.dataset_exists():
            print(f"Test dataset already exists at {self.final_output_path}")
        else:
            print("Test dataset not found. Checking source datasets...")
            crowdflower_exists, cbet_exists = self.ensure_datasets()
            
            if crowdflower_exists and cbet_exists:
                print("Starting dataset fusion...")
                self._create_dataset()
                print("Fusion complete.")
            else:
                print("Unable to create test dataset due to missing source datasets.")
    
    def get_dataset(self) -> pd.DataFrame:
        """Get the fused dataset as a DataFrame."""
        self.ensure_dataset()
        return pd.read_csv(self.final_output_path)
    
    def get_label_column(self) -> str:
        return "emotion"
    
    def get_text_column(self) -> str:
        return "text"