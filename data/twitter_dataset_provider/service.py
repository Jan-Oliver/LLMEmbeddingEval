import csv
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from ..abstract_dataset_provider import AbstractDatasetProvider
from .tweet_translator import TweetTranslator


class TwitterDatasetProvider(AbstractDatasetProvider):

    """
    High-level service to check for, download, fuse, translate,
    and provide access to Twitter emotion datasets.
    """

    # URLs for the datasets
    CROWDFLOWER_URL = "https://raw.githubusercontent.com/tlkh/text-emotion-classification/master/dataset/original/text_emotion.csv"
    CBET_URL = "https://webdocs.cs.ualberta.ca/~zaiane/CBET/CBET.csv"


    def __init__(
        self,
        output_path: Path,
        target_emotions: List[str],
        openai_api_key: str,
        model: str = "gpt-4.1-mini",
        balance_emotions: bool = True,
        max_workers: int = 5,
        debug: bool = False
    ) -> None:
        """
        Initialize the TwitterDatasetProvider.

        Args:
            output_path: Path where the final fused dataset will be saved
            target_emotions: List of emotion categories to keep and map to
            openai_api_key: OpenAI API key for translation
            model: OpenAI model to use for translation
            target_language: The language to translate texts into (e.g., "German")
            balance_emotions: Whether to balance the dataset by emotion category
            max_workers: Maximum number of concurrent workers for API calls
            debug: Whether to print debug information
        """
        super().__init__(output_path)
        self.output_path = output_path.joinpath("twitter_dataset")
        self.crowdflower_path = self.output_path / "crowdflower.csv"
        self.cbet_path = self.output_path / "cbet.csv"

        # Paths for intermediate and final datasets
        self.english_fused_path = self.output_path / "twitter_fused_en.csv" # Fused English dataset
        self.translated_fused_path = self.output_path / "twitter_fused_translated_de.csv" # Fused + Translation intermediate
        # Corrected the final output path name
        self.final_output_path = self.output_path / "twitter_emotion_dataset_combined.csv" # Final combined EN/DE dataset

        self.balance_emotions = balance_emotions
        self.target_emotions = target_emotions
        self.openai_api_key = openai_api_key
        self.model = model
        self.max_workers = max_workers
        self.debug = debug


        # Create data directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Instantiate translator (passing parameters)
        self.translator = TweetTranslator(
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
            # Attempt to read with utf-8, fallback to latin-1 if needed
            try:
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip', quoting=csv.QUOTE_ALL, escapechar='\\')
            except Exception as e:
                self._debug_print(f"Error reading CSV {path} with utf-8: {e}. Attempting with latin-1.")
                df = pd.read_csv(path, encoding='latin-1', on_bad_lines='skip', quoting=csv.QUOTE_ALL, escapechar='\\')

            if df.empty:
                self._debug_print(f"File is empty: {path}")
                return None
            # Basic check to see if it's a dataframe with columns
            if df.shape[1] == 0:
                self._debug_print(f"File loaded but has no columns: {path}")
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
        # Check for required columns in the final combined dataset
        required_cols = ['content', 'language', 'emotion', 'source']
        if not all(col in df.columns for col in required_cols):
            self._debug_print(f"Final dataset {self.final_output_path} missing required columns: {required_cols}.")
            return False
        # Check for presence of both languages
        if 'language' not in df.columns: # Redundant check, but safe
             return False
        lang_counts = df['language'].value_counts()
        has_en = 'en' in lang_counts and lang_counts['en'] > 0
        has_de = 'de' in lang_counts and lang_counts['de'] > 0
        self._debug_print(f"Final dataset language counts: {lang_counts}. Has EN: {has_en}, Has DE: {has_de}")
        return has_en and has_de


    def dataset_exists(self) -> bool:
        """Check if the final combined processed dataset already exists."""
        return self._check_combined_dataset_exists()

    def _download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a file from a given URL.

        Args:
            url: URL to download from
            output_path: Path to save the downloaded file

        Returns:
            bool: True if download was successful, False otherwise
        """
        if output_path.exists():
            # Check if file is empty, if so, try downloading again
            try:
                 if output_path.stat().st_size > 0:
                    print(f"File already exists and is not empty: {output_path}")
                    return True
                 else:
                    print(f"File exists but is empty: {output_path}. Redownloading.")
            except Exception as e:
                 self._debug_print(f"Error checking file size {output_path}: {e}. Redownloading.")


        try:
            print(f"Downloading {url} to {output_path}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status() # Raise an exception for HTTP errors

            # Write content as text, specifying encoding
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)

            print(f"Successfully downloaded {url}")
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

    def ensure_datasets(self) -> Tuple[bool, bool]:
        """
        Ensure both raw source datasets are available, downloading them if needed.

        Returns:
            Tuple[bool, bool]: Tuple indicating if CrowdFlower and CBET datasets are available
        """
        # This method downloads the *raw* source files
        crowdflower_available = self._download_file(self.CROWDFLOWER_URL, self.crowdflower_path)
        cbet_available = self._download_file(self.CBET_URL, self.cbet_path)

        # Note: This method only guarantees *download*, not processing success
        return crowdflower_available, cbet_available


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

        df = pd.DataFrame() # Initialize empty DataFrame
        try:
            # Read the downloaded CSV
            df = pd.read_csv(self.crowdflower_path, encoding='utf-8', on_bad_lines='skip')

            self._debug_print(f"CrowdFlower raw columns: {df.columns.tolist()}")

            # Confirm we have the expected columns
            if 'content' not in df.columns or 'sentiment' not in df.columns:
                print(f"Warning: Expected columns 'content' or 'sentiment' not found in CrowdFlower dataset at {self.crowdflower_path}")
                self._debug_print(f"Available columns: {df.columns.tolist()}")
                return pd.DataFrame() # Return empty if essential columns are missing

            # Map columns to our standard format
            df = df.rename(columns={
                'content': 'text',
                'sentiment': 'source_emotion' # Keep original sentiment for mapping
            })

            # Add source column
            df['source'] = 'crowdflower'

            # Create reverse mapping from CrowdFlower emotions to target emotions
            reverse_mapping = {}
            for target, sources in CROWDFLOWER_MAPPINGS.items():
                for source in sources:
                    reverse_mapping[source.lower()] = target

            # Map to target emotions
            df['emotion'] = df['source_emotion'].str.lower().map(reverse_mapping)

            # Drop rows with emotions we don't want to map (NaN in 'emotion')
            df = df.dropna(subset=['emotion'])

            # Keep only necessary columns for the fused dataset structure
            df = df[['text', 'emotion', 'source']]

            print(f"Processed CrowdFlower dataset: {len(df)} rows")
            if not df.empty:
                 self._debug_print(f"Processed CrowdFlower columns: {df.columns.tolist()}")
                 self._debug_print(f"CrowdFlower emotion distribution:\n{df['emotion'].value_counts()}")

            return df
        except FileNotFoundError:
            print(f"CrowdFlower raw data not found at {self.crowdflower_path}. Cannot process.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error processing CrowdFlower dataset: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()


    def _process_cbet(self) -> pd.DataFrame:
        """
        Process the CBET dataset.

        Returns:
            pd.DataFrame: Processed CBET data with mapped target emotions
        """
        df = pd.DataFrame() # Initialize empty DataFrame
        try:
            # Read the downloaded CSV
            # CBET often uses Latin-1 encoding
            df = pd.read_csv(self.cbet_path, encoding='latin-1', on_bad_lines='skip')

            self._debug_print(f"CBET raw columns: {df.columns.tolist()}")

            # Ensure 'text' column exists (it might be called 'sentence' or something else)
            text_col = 'text'
            if text_col not in df.columns:
                if 'sentence' in df.columns:
                    df = df.rename(columns={'sentence': text_col})
                # If there's no standard text column, try to infer or handle missing data
                else:
                     print(f"Warning: Standard text/sentence column not found in CBET dataset at {self.cbet_path}")
                     self._debug_print(f"Available columns: {df.columns.tolist()}")
                     # Depending on structure, might need more complex logic
                     # For now, return empty if no identifiable text column
                     return pd.DataFrame()

            # The CBET dataset has intensity scores for each emotion in separate columns
            # We need to determine the dominant emotion for each text

            # Mapping from CBET emotion columns to our target emotions
            # Adjusted mapping based on common CBET columns and target emotions
            emotion_col_mapping = {
                'anger': 'Anger',
                'joy': 'Joy',
                'sadness': 'Sadness',
                'fear': 'Fear',
                'love': 'Joy',
                'thankfulness': 'Joy',
            }

            # Filter mapping to include only target emotions
            emotion_col_mapping = {k: v for k, v in emotion_col_mapping.items() if v in self.target_emotions}

            # Check if we have the expected emotion columns in the DataFrame
            found_emotion_cols = [col for col in emotion_col_mapping.keys() if col in df.columns]
            if not found_emotion_cols:
                print(f"Warning: No relevant emotion intensity columns found in CBET dataset at {self.cbet_path}")
                self._debug_print(f"Looking for any of: {list(emotion_col_mapping.keys())}")
                self._debug_print(f"Available columns: {df.columns.tolist()}")
                # If no emotion columns, we can't assign labels, return empty
                return pd.DataFrame()


            # Create a new DataFrame to store the results
            result_rows = []

            # Process each row
            for _, row in df.iterrows():
                text = row[text_col]

                # Get the emotion scores for the emotions we care about
                emotion_scores = {}
                for cbet_emotion_col, target_emotion in emotion_col_mapping.items():
                    # Check if column exists AND value is numeric
                    if cbet_emotion_col in row and pd.api.types.is_numeric_dtype(type(row[cbet_emotion_col])):
                         current_score = emotion_scores.get(target_emotion, 0.0)
                         try:
                             new_score = float(row[cbet_emotion_col])
                             emotion_scores[target_emotion] = max(current_score, new_score)
                         except (ValueError, TypeError):
                             # Ignore non-numeric scores
                             self._debug_print(f"Warning: Non-numeric score '{row[cbet_emotion_col]}' for {cbet_emotion_col} in CBET row. Skipping.")
                             pass # Skip this score


                # Determine if the text is neutral (all emotion scores are 0 or missing)
                # Only consider scores for emotions *we mapped and found columns for*
                if not emotion_scores: # If no scores were found/mapped
                     all_relevant_scores_zero = True
                else:
                     all_relevant_scores_zero = all(score == 0.0 for score in emotion_scores.values())


                # If all relevant scores are zero AND Neutral is a target emotion, assign Neutral
                dominant_emotion = None
                max_score = -1.0 # Use -1.0 to handle case where all scores are exactly 0
                dominant_emotion_candidates = [] # In case of ties

                for emotion, score in emotion_scores.items():
                    if score > max_score:
                        max_score = score
                        dominant_emotion_candidates = [emotion] # Start a new list of candidates
                    elif score == max_score and max_score > 0: # Add to candidates if it's a tie and score is > 0
                         dominant_emotion_candidates.append(emotion)

                if max_score <= 0 and 'Neutral' in self.target_emotions:
                    dominant_emotion = 'Neutral'
                elif len(dominant_emotion_candidates) == 1:
                    dominant_emotion = dominant_emotion_candidates[0]
                elif len(dominant_emotion_candidates) > 1:
                    # Handle ties: pick one deterministically (e.g., first alphabetically)
                    dominant_emotion = sorted(dominant_emotion_candidates)[0]


                # Only add rows with an identified dominant emotion that is one of our targets
                if dominant_emotion in self.target_emotions:
                    result_rows.append({
                        'text': text,
                        'emotion': dominant_emotion,
                        'source': 'cbet',
                    })
                elif dominant_emotion is None and not all_relevant_scores_zero:
                     # This case indicates scores > 0 but no dominant emotion was picked (e.g., mapping issue)
                     self._debug_print(f"Warning: CBET row with scores > 0 but no dominant emotion picked: {row.to_dict()}")


            # Create DataFrame from processed rows
            result_df = pd.DataFrame(result_rows)

            # Keep only necessary columns
            if not result_df.empty:
                result_df = result_df[['text', 'emotion', 'source']]
                print(f"Processed CBET dataset: {len(result_df)} rows")
                self._debug_print(f"Processed CBET columns: {result_df.columns.tolist()}")
                self._debug_print(f"CBET emotion distribution:\n{result_df['emotion'].value_counts()}")
            else:
                print("Warning: No valid rows extracted from CBET dataset")

            return result_df
        except FileNotFoundError:
            print(f"CBET raw data not found at {self.cbet_path}. Cannot process.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error processing CBET dataset: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()


    def _fuse_datasets(self) -> pd.DataFrame:
        """
        Fuse the processed CrowdFlower and CBET datasets into a single DataFrame.
        This produces the initial English, labeled dataset.

        Returns:
            pd.DataFrame: Fused dataset with consistent target emotions
        """
        print("Fusing processed datasets...")
        # Process individual datasets - they already map to target emotions
        crowdflower_df = self._process_crowdflower()
        cbet_df = self._process_cbet()

        print(f"CrowdFlower rows after processing: {len(crowdflower_df)}")
        print(f"CBET rows after processing: {len(cbet_df)}")

        # Combine datasets
        combined_df = pd.concat([crowdflower_df, cbet_df], ignore_index=True) # Combine both DFs

        # Ensure we have only our target emotions (should already be handled by processing, but good check)
        valid_emotions = set(self.target_emotions)
        if not combined_df.empty:
            combined_df = combined_df[combined_df['emotion'].isin(valid_emotions)]
            print(f"Total fused rows before balancing/shuffling: {len(combined_df)}")
        else:
             print("Fused dataset is empty after processing.")

        return combined_df


    def _save_and_load_english_fused(self) -> pd.DataFrame:
        """
        Fuses raw data, balances/shuffles, adds language='en',
        saves to intermediate file, and returns the DataFrame.
        Loads from file if it already exists.
        """
        df_english_fused = self._check_file_exists_and_is_dataframe(self.english_fused_path)

        if df_english_fused is not None:
            # Basic check: does it have 'text', 'emotion', 'language'='en'?
            required_cols = ['text', 'emotion', 'language']
            if all(col in df_english_fused.columns for col in required_cols):
                 if 'en' in df_english_fused['language'].unique():
                    print(f"English fused data already exists and is valid at {self.english_fused_path}. Skipping fusion/processing.")
                    return df_english_fused
                 else:
                    print(f"Warning: English fused data file {self.english_fused_path} exists but has no English rows. Refusing/reprocessing.")
                    df_english_fused = None # Force reprocessing
            else:
                print(f"Warning: English fused data file {self.english_fused_path} missing expected columns. Refusing/reprocessing.")
                df_english_fused = None # Force reprocessing


        if df_english_fused is None:
            # Step 1: Fuse the processed source datasets
            fused_df = self._fuse_datasets()

            if fused_df.empty:
                print("Fusion resulted in an empty dataset. Cannot save English fused data.")
                return pd.DataFrame()

            # Step 2: Balance the dataset if required
            if self.balance_emotions:
                before_balance = len(fused_df)
                fused_df = self._balance_dataset(fused_df)
                after_balance = len(fused_df)
                print(f"Dataset balanced: {before_balance} samples -> {after_balance} samples")

            # Step 3: Add language column
            fused_df['language'] = 'en'

            # Step 4: Shuffle the dataset
            fused_df = fused_df.sample(frac=1, random_state=42).reset_index(drop=True)

            # Save the English fused dataset
            if not fused_df.empty:
                 fused_df.to_csv(self.english_fused_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
                 print(f"English fused dataset saved to {self.english_fused_path}")
            else:
                 print(f"English fused dataset is empty, not saving {self.english_fused_path}")

            df_english_fused = fused_df


        return df_english_fused
    
    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset to have an equal number of samples per emotion.

        Args:
            df: DataFrame with 'emotion' column

        Returns:
            pd.DataFrame: Balanced DataFrame
        """
        
        if not self.balance_emotions or df.empty or 'emotion' not in df.columns:
            if self.balance_emotions:
                 if df.empty: print("Warning: Cannot balance an empty dataset.")
                 elif 'emotion' not in df.columns: print("Warning: Cannot balance dataset, 'emotion' column missing.")
            return df

        emotion_counts = df['emotion'].value_counts()

        if emotion_counts.empty:
            print("Warning: No emotions found in dataset for balancing.")
            return df

        min_count = emotion_counts.min()

        if min_count == 0:
            print("Warning: Minimum emotion count is 0. Balancing cannot create samples for all classes.")
            return df

        print(f"Balancing dataset by undersampling majority classes to {min_count} samples per emotion.")

        balanced_dfs = []
        for emotion in self.target_emotions:
            if emotion in emotion_counts:
                emotion_df = df[df['emotion'] == emotion]
                sampled_df = emotion_df.sample(n=min_count, random_state=42)
                balanced_dfs.append(sampled_df)

        if not balanced_dfs:
            print("Warning: Balancing resulted in an empty list of DataFrames.")
            return pd.DataFrame()

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)

        return balanced_df
    
    def _translate_fused_data(self, df_english_fused: pd.DataFrame) -> pd.DataFrame:
        """
        Translates the 'text' column of the English fused data and adds
        translation results as new columns.
        Loads from intermediate file if it already exists.

        Args:
            df_english_fused: DataFrame containing the English fused data with 'text' and 'emotion' columns.
        """
        df_translated = self._check_file_exists_and_is_dataframe(self.translated_fused_path)

        if df_translated is not None:
            # Basic check: does it have 'translated_text' and original columns?
            required_cols = ['text', 'emotion', 'translated_text']
            if all(col in df_translated.columns for col in required_cols):
                print(f"Translated data already exists and is valid at {self.translated_fused_path}. Skipping translation.")
                return df_translated
            else:
                print(f"Warning: Translated data file {self.translated_fused_path} seems incomplete. Retranslating.")
                df_translated = None # Force retranslation

        if df_translated is None:
            print(f"Translating 'text' column to German...")

            # Ensure input has the text column
            if 'text' not in df_english_fused.columns:
                 raise ValueError("Input DataFrame to _translate_fused_data must contain 'text' column.")


            # Translate the 'text' column using the translator
            # The translator adds 'translated_text' and other translation result columns
            # It keeps all existing columns from df_english_fused ('text', 'emotion', 'source', 'language')
            df_translated = self.translator.translate_dataframe(
                df_english_fused.copy(), # Work on a copy
                text_column='text' # Translate the 'text' column
            )

            # Save this intermediate DataFrame (contains text, translated_text, emotion, source, language='en', plus translator metadata)
            if not df_translated.empty:
                 df_translated.to_csv(self.translated_fused_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
                 print(f"Translated data saved to {self.translated_fused_path}")
            else:
                 print(f"Translated dataset is empty, not saving {self.translated_fused_path}")


        return df_translated


    def _combine_translated_data(self, df_with_translations_and_labels: pd.DataFrame) -> pd.DataFrame:
        """
        Combine English and German data into a single DataFrame with 'language' column
        and a unified 'content' column by duplicating rows and restructuring columns.

        Args:
            df_with_translations_and_labels: DataFrame containing English text ('text'),
                                             its translation ('translated_text'),
                                             and labels ('emotion', 'source', 'language'='en', etc.).
        """
        # Check if the final dataset already exists with both languages
        if self._check_combined_dataset_exists():
            print(f"Final combined dataset already exists at {self.final_output_path}. Skipping combination.")
            return pd.read_csv(self.final_output_path)

        print("Combining English and German datasets by duplicating rows...")

        # Ensure the input DataFrame has necessary columns
        required_cols = ['text', 'translated_text', 'emotion', 'source', 'language'] # Basic check
        if not all(col in df_with_translations_and_labels.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df_with_translations_and_labels.columns]
             raise ValueError(f"Input DataFrame to _combine_translated_data missing required columns: {missing}")

        # Identify the columns that contain labels and source info that should be copied
        # These are 'emotion' and 'source'. 'language' will be handled separately.
        label_and_source_cols = ['emotion', 'source']


        # --- Prepare English rows for the final output ---
        # Select 'text' as content, the label/source columns, and the original language
        df_english_final = df_with_translations_and_labels[label_and_source_cols + ['text', 'language']].copy()
        df_english_final = df_english_final.rename(columns={'text': 'content'})
        # Ensure language is explicitly set to 'en' for these rows
        df_english_final['language'] = 'en'


        # --- Prepare German rows for the final output ---
        # Select 'translated_text' as content, and copy the SAME label/source columns
        df_german_final = df_with_translations_and_labels[label_and_source_cols + ['translated_text']].copy()
        df_german_final = df_german_final.rename(columns={'translated_text': 'content'})
        # Set language column for German rows
        df_german_final['language'] = 'de'


        # --- Concatenate the English and German DataFrames ---
        # Ensure both DataFrames have the same columns before concatenating.
        # The required columns for the final dataset are 'content', 'language', 'emotion', 'source'.
        final_column_order = ['content', 'language'] + label_and_source_cols

        # Ensure order is consistent (should be if defined carefully above, but safe)
        df_english_final = df_english_final[final_column_order]
        df_german_final = df_german_final[final_column_order]

        # Concatenate them
        final_df = pd.concat([df_english_final, df_german_final], ignore_index=True)

        # Final check on columns (optional)
        final_df = final_df.loc[:,~final_df.columns.duplicated()].copy()


        # Save the final combined dataset
        if not final_df.empty:
            final_df.to_csv(self.final_output_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
            print(f"Final combined dataset saved to {self.final_output_path}")
            print("\nFinal Combined Dataset Statistics:")
            print(f"Total samples: {len(final_df)}")
            if 'language' in final_df.columns:
                print("\nSamples per language:")
                print(final_df['language'].value_counts())
            if 'emotion' in final_df.columns:
                print("\nSamples per emotion category:")
                # Group by language and emotion for detailed stats
                print(final_df.groupby('language')['emotion'].value_counts())
            if 'source' in final_df.columns:
                 print("\nSamples per source:")
                 print(final_df.groupby('language')['source'].value_counts())


        else:
            print(f"Final combined dataset is empty, not saving {self.final_output_path}")


        return final_df


    def ensure_dataset(self) -> None:
        """Ensure the combined processed and translated dataset exists, creating it if necessary."""
        # Check the final state first
        if self._check_combined_dataset_exists():
            print(f"Twitter combined dataset already exists at {self.final_output_path}. All steps are complete.")
            return

        print("Twitter combined dataset not found. Starting creation process...")

        # Step 1: Download Raw Source Files
        # This checks if raw data exists and is valid, downloads if not
        crowdflower_available, cbet_available = self.ensure_datasets()
        if not crowdflower_available or not cbet_available:
            print("Failed to download one or both raw source datasets. Cannot proceed.")
            return

        # Step 2: Load, Process, Fuse, Balance, Shuffle English Data
        # This checks if english_fused_path exists, loads/creates if not.
        # It calls _fuse_datasets internally if needed.
        df_english_fused = self._save_and_load_english_fused()
        if df_english_fused.empty:
            print("Processing and fusing resulted in an empty English dataset. Cannot proceed.")
            return


        # Step 3: Translate Fused English Data
        # This checks if translated_fused_path exists, loads/creates if not.
        # It calls the TweetTranslator internally if needed.
        df_translated = self._translate_fused_data(df_english_fused)
        if df_translated.empty:
             print("Translation step resulted in an empty dataset. Cannot proceed.")
             return


        # Step 4: Combine English and German Datasets
        # This checks if the final combined dataset exists, loads/creates if not.
        # It restructures the translated data into the final combined format.
        self._combine_translated_data(df_translated)

        print("Dataset creation process complete.")


    def get_dataset(self) -> pd.DataFrame:
        """Get the combined processed and translated dataset as a DataFrame."""
        self.ensure_dataset()
        df = self._check_file_exists_and_is_dataframe(self.final_output_path)
        if df is None:
            # This should ideally not happen if ensure_dataset finished successfully,
            # but as a safeguard.
            raise FileNotFoundError(f"Final dataset not found or is empty after creation attempt: {self.final_output_path}")
        return df

    def get_label_column(self) -> str:
        """Get the name of the column that contains the emotion labels."""
        # The emotion label column is the same for both English and German rows
        return "emotion"

    def get_text_column(self) -> str:
        """
        Get the name of the column that contains the text data (original English
        or translated German).
        """
        # In the final combined dataset, the text column is named 'content'
        return "content"