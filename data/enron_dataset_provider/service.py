import csv
import os
import io
import zipfile
import requests
import pandas as pd
from pathlib import Path
from typing import List, Optional, Any
from tqdm import tqdm
import re # Import re for checking empty dataframes

# Assuming your file structure allows these imports
from ..abstract_dataset_provider import AbstractDatasetProvider
from .email_preprocessor import EnronEmailProcessor
from .email_classifier import EnronEmailClassifier
from .email_translator import EmailTranslator


class EnronDatasetProvider(AbstractDatasetProvider):
    """
    High-level service to check for, preprocess, classify, translate,
    and provide access to the Enron email dataset.
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
        debug: bool = False
    ) -> None:
        """
        Initialize the EnronDatasetProvider.

        Args:
            output_path: Path where the processed dataset will be saved
            target_emotions: List of emotion labels to classify
            openai_api_key: OpenAI API key
            target_size: Number of emails to sample from the dataset
            num_classifications: Number of times to classify each email
            max_workers: Maximum number of concurrent workers for API calls
            model: OpenAI model to use for classification and translation
            debug: Whether to print debug information
        """
        super().__init__(output_path)
        # Define all file paths for different stages
        self.raw_data_path = output_path.joinpath("enron", "raw_data.csv")
        self.preprocessed_data_path = output_path.joinpath("enron", "enron_email_preprocessed.csv") # New intermediate file
        self.english_classified_path = output_path.joinpath("enron", "enron_email_classified_en.csv")
        self.translated_data_path = output_path.joinpath("enron", f"enron_email_translated_de.csv") # Intermediate translated file
        self.final_output_path = output_path.joinpath("enron", "enron_email_emotions.csv") # This will contain combined en/de data

        self.target_size = target_size
        self.target_emotions = target_emotions
        self.num_classifications = num_classifications
        self.max_workers = max_workers
        self.openai_api_key = openai_api_key
        self.enron_dataset_path = "https://olli-master-thesis.s3.eu-west-1.amazonaws.com/emails.csv.zip"
        self.model = model
        self.debug = debug

        # Create data directory if it doesn't exist
        self.final_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize processors/classifiers/translators
        self.preprocessor = EnronEmailProcessor(debug=self.debug)
        self.classifier = EnronEmailClassifier(
            target_emotions=self.target_emotions,
            num_classifications=self.num_classifications,
            max_workers=self.max_workers,
            model=self.model,
            api_key=self.openai_api_key,
            debug=self.debug
        )
        self.translator = EmailTranslator(
            max_workers=self.max_workers,
            model=self.model,
            api_key=self.openai_api_key,
            debug=self.debug
        )

    def _debug_print(self, message: str, data: Any = None) -> None:
        """Print debug information if debug mode is enabled."""
        if not self.debug:
            return
        print(f"[DEBUG] {message}")
        if data is not None:
            print(data)
            print("-" * 40)


    def _check_file_exists_and_is_dataframe(self, path: Path) -> Optional[pd.DataFrame]:
        """Checks if a file exists, loads it, and returns DataFrame if successful and not empty."""
        if not path.exists():
            self._debug_print(f"File not found: {path}")
            return None
        try:
            df = pd.read_csv(path)
            if df.empty:
                 self._debug_print(f"File is empty: {path}")
                 return None
            self._debug_print(f"File loaded successfully: {path} ({len(df)} rows)")
            return df
        except Exception as e:
            self._debug_print(f"Error loading file {path}: {e}")
            return None

    def _check_combined_dataset_exists(self) -> bool:
        """Checks if the final combined dataset exists, is not empty, and contains both languages."""
        df = self._check_file_exists_and_is_dataframe(self.final_output_path)
        if df is None:
            return False
        if 'language' not in df.columns:
            self._debug_print(f"Final dataset {self.final_output_path} missing 'language' column.")
            return False
        lang_counts = df['language'].value_counts()
        has_en = 'en' in lang_counts and lang_counts['en'] > 0
        has_de = 'de' in lang_counts and lang_counts['de'] > 0
        self._debug_print(f"Final dataset language counts: {lang_counts}. Has EN: {has_en}, Has DE: {has_de}")
        return has_en and has_de


    def dataset_exists(self) -> bool:
        """Check if the final combined processed dataset already exists."""
        return self._check_combined_dataset_exists()

    def _download_and_extract_data(self) -> None:
        """
        Download the zipped CSV file from S3 and extract it.
        Raises exception if fails.
        """
        if self._check_file_exists_and_is_dataframe(self.raw_data_path) is not None:
            print(f"Raw data already exists and is valid at {self.raw_data_path}")
            return

        try:
            print(f"Downloading data from {self.enron_dataset_path}...")
            response = requests.get(str(self.enron_dataset_path), stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading raw data")

            zip_content = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    zip_content.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()

            zip_content.seek(0)
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

            print("Extracting zip file...")
            with zipfile.ZipFile(zip_content) as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if not csv_files:
                    raise Exception("No CSV files found in the zip archive.")

                csv_file = csv_files[0]
                print(f"Extracting {csv_file} to {self.raw_data_path}...")
                with zip_ref.open(csv_file) as zf, open(self.raw_data_path, 'wb') as f:
                    f.write(zf.read())

            print(f"Successfully downloaded and extracted raw data.")

        except Exception as e:
            raise Exception(f"Failed to download or extract raw dataset: {e}") from e

    def _load_and_preprocess_emails(self) -> pd.DataFrame:
        """Load and preprocess the Enron emails."""
        df_preprocessed = self._check_file_exists_and_is_dataframe(self.preprocessed_data_path)
        if df_preprocessed is not None:
            print(f"Preprocessed data already exists and is valid at {self.preprocessed_data_path}. Skipping preprocessing.")
            # Ensure it has the expected columns if loading from file
            expected_cols = ['message', 'preprocessed_text']
            if not all(col in df_preprocessed.columns for col in expected_cols):
                 print(f"Warning: Preprocessed data file {self.preprocessed_data_path} missing expected columns. Reprocessing.")
                 df_preprocessed = None # Force reprocessing

        if df_preprocessed is None:
            print(f"Loading up to {self.target_size} emails from {self.raw_data_path} for preprocessing...")

    
            df_raw = pd.read_csv(self.raw_data_path, nrows=self.target_size)

            print("Preprocessing emails...")
            tqdm.pandas()
            # Apply preprocessing to the 'message' column (corrected from 'text')
            df_raw['preprocessed_text'] = df_raw['message'].progress_apply(self.preprocessor.preprocess_email)

            # Filter out empty messages (after preprocessing)
            df_filtered = df_raw[df_raw['preprocessed_text'] != ""].reset_index(drop=True)

            # Drop duplicate rows based on preprocessed text
            df_filtered = df_filtered.drop_duplicates(subset='preprocessed_text')

            print(f"Loaded {len(df_raw)} raw emails, {len(df_filtered)} remained after filtering and removing duplicates.")

            df_preprocessed = df_filtered[['preprocessed_text']]

            # Save the preprocessed data
            df_preprocessed.to_csv(self.preprocessed_data_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
            print(f"Preprocessed dataset saved to {self.preprocessed_data_path}")


        return df_preprocessed


    def _assign_emotion_labels(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        """
        Assign emotion labels to the emails using the EnronEmailClassifier.
        Adds the 'language' column initialized to 'en'.
        """
        df_classified_en = self._check_file_exists_and_is_dataframe(self.english_classified_path)
        if df_classified_en is not None:
             # Basic check: does it have 'language' and 'emotion'?
             if 'language' in df_classified_en.columns and 'emotion' in df_classified_en.columns and not df_classified_en[df_classified_en['language']=='en'].empty:
                print(f"English classified data already exists and is valid at {self.english_classified_path}. Skipping classification.")
                return df_classified_en # Return the loaded DF
             else:
                 print(f"Warning: English classified data file {self.english_classified_path} seems incomplete. Reclassifying.")
                 df_classified_en = None # Force reclassification

        if df_classified_en is None:
            print("Classifying email emotions with GPT API...")

            # Classify the preprocessed emails
            classified_df = self.classifier.classify_emails(df_preprocessed.copy(), text_column='preprocessed_text') # Classify using preprocessed_text

            # Rename 'final_emotion' to 'emotion' to be consistent
            classified_df = classified_df.rename(columns={'final_emotion': 'emotion'})

            # Add the language column for English data
            classified_df['language'] = 'en'

            # Save the English classified results
            classified_df.to_csv(self.english_classified_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
            print(f"English classified dataset saved to {self.english_classified_path}")

            df_classified_en = classified_df # Set the result DF


        return df_classified_en


    def _translate_emails(self, df_classified_en: pd.DataFrame) -> pd.DataFrame:
        """
        Translate the preprocessed email text and add the translations as a new column.
        """
        df_translated = self._check_file_exists_and_is_dataframe(self.translated_data_path)
        if df_translated is not None:
            # Basic check: does it have 'translated_text'?
            if 'translated_text' in df_translated.columns:
                print(f"Translated data already exists and is valid at {self.translated_data_path}. Skipping translation.")
                return df_translated # Return the loaded DF
            else:
                print(f"Warning: Translated data file {self.translated_data_path} missing 'translated_text'. Retranslating.")
                df_translated = None # Force retranslation


        if df_translated is None:
            print(f"Translating preprocessed email text to German...")

            # Translate the 'preprocessed_text' column
            df_translated = self.translator.translate_emails_df(
                df_classified_en.copy(), # Translate based on the English classified DF
                text_column='preprocessed_text' # Translate the preprocessed text
            )

            df_translated['language'] = 'de'
            
            # Save this intermediate DataFrame
            df_translated.to_csv(self.translated_data_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
            print(f"Translated data saved to {self.translated_data_path}")

        return df_translated

    def _combine_datasets(self, df_translated: pd.DataFrame) -> pd.DataFrame:
        """
        Combine English and German data into a single DataFrame with 'language' column
        and a unified 'content' column.
        """
        if self._check_combined_dataset_exists():
            print(f"Final combined dataset already exists at {self.final_output_path}. Skipping combination.")
            return pd.read_csv(self.final_output_path)

        print("Combining English and German datasets...")

        # Define the common columns needed in the final dataset (labels + content + language)
        # We'll keep the classification columns as is
        label_columns = [
             col for col in df_translated.columns
             if col.startswith('emotion') or col in ['confidence', 'emotion'] # Include 'emotion' which is final_emotion renamed
        ]

        # Ensure 'emotion' is included if it was renamed
        if 'emotion' not in label_columns and 'final_emotion' in df_translated.columns:
             label_columns.append('final_emotion') # Should have been renamed by _assign_emotion_labels, but as safety

        # --- Create English rows ---
        df_en = df_translated.copy()
        # Use preprocessed_text as content for English rows
        df_en = df_en.rename(columns={'preprocessed_text': 'content'})
        # Ensure the language column is correct (should be 'en' from _assign_emotion_labels)
        df_en['language'] = 'en'
        # Select only the final desired columns
        df_en = df_en[['content', 'language'] + label_columns]


        # --- Create German rows ---
        # Create a copy based on the English data structure
        df_de = df_translated.copy()
        # Replace 'preprocessed_text'/'content' with 'translated_text'
        df_de = df_de.rename(columns={'translated_text': 'content'})
        # Set language column for German rows
        df_de['language'] = 'de'
        # Select only the final desired columns (classification labels are copied from the original row)
        df_de = df_de[['content', 'language'] + label_columns]

        # Concatenate the English and German DataFrames
        final_df = pd.concat([df_en, df_de], ignore_index=True)

        # Optional: Clean up columns if any duplicates were introduced or order
        final_df = final_df.loc[:,~final_df.columns.duplicated()].copy() # Drop potential duplicates introduced by selection/renaming

        # Ensure 'emotion' is present in the final column list
        if 'emotion' not in final_df.columns and 'final_emotion' in final_df.columns:
             final_df = final_df.rename(columns={'final_emotion': 'emotion'})

        # Reorder columns for clarity (optional)
        final_column_order = ['content', 'language', 'emotion', 'confidence'] + [col for col in final_df.columns if col.startswith('emotion_attempt_') or col == 'emotion_counts']
        # Filter final_column_order to only include columns actually present in final_df
        final_column_order = [col for col in final_column_order if col in final_df.columns]
        final_df = final_df[final_column_order]


        # Save the final combined dataset
        final_df.to_csv(self.final_output_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"Final combined dataset saved to {self.final_output_path}")

        return final_df


    def ensure_dataset(self) -> None:
        """Ensure the processed and translated dataset exists, creating it if necessary."""
        if self._check_combined_dataset_exists():
            print(f"Enron combined dataset already exists at {self.final_output_path}. All steps are complete.")
            return

        print("Enron combined dataset not found. Starting creation process...")

        # Step 1: Download and Extract Raw Data
        self._download_and_extract_data()

        # Step 2: Load, Preprocess Emails
        # This loads raw data, preprocesses, and saves to preprocessed_data_path
        preprocessed_df = self._load_and_preprocess_emails()
        if preprocessed_df.empty:
             print("Preprocessing resulted in an empty dataset. Cannot proceed.")
             return

        # Step 3: Classify English Emails
        # This loads preprocessed data, classifies, adds language='en', and saves to english_classified_path
        # It returns the loaded/created English classified DataFrame
        classified_en_df = self._assign_emotion_labels(preprocessed_df)
        if classified_en_df.empty:
             print("Classification resulted in an empty dataset. Cannot proceed.")
             return # Stop if no data


        # Step 4: Translate Preprocessed English Emails
        # This loads English classified data, translates preprocessed_text, adds translated_text, and saves to translated_data_path
        df_translated = self._translate_emails(classified_en_df)
        if df_translated.empty:
             print("Translation step resulted in an empty dataset. Cannot proceed.")
             return # Stop if no data


        # Step 5: Combine English and German Datasets
        # This loads translated data and creates the final combined dataset
        self._combine_datasets(df_translated)

        print("Dataset creation process complete.")


    def get_dataset(self) -> pd.DataFrame:
        """Get the processed and translated dataset as a DataFrame."""
        self.ensure_dataset() # Ensure dataset is fully created
        df = self._check_file_exists_and_is_dataframe(self.final_output_path)
        if df is None:
             raise FileNotFoundError(f"Final dataset not found or is empty after creation attempt: {self.final_output_path}")
        return df

    def get_label_column(self) -> str:
        """Get the name of the column that contains the emotion labels."""
        # The emotion label column is the same for both English and German rows
        return "emotion"

    def get_text_column(self) -> str:
        """
        Get the name of the column that contains the text data (preprocessed English
        or translated German).
        """
        # In the final combined dataset, the text column is named 'content'
        return "content"