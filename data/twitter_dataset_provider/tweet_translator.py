import openai
import time
import concurrent.futures
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any, Optional

class TweetTranslator:
    """
    Translates text (like tweets) using GPT.
    """

    def __init__(
        self,
        max_workers: int = 5,
        model: str = "gpt-4.1-mini", # Using a placeholder model name
        temperature: float = 0.5,
        api_key: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize the tweet translator.

        Args:
            target_language: The language to translate into (e.g., "German")
            max_workers: Maximum number of concurrent workers for API calls
            model: GPT model to use
            temperature: Temperature for the GPT model
            api_key: OpenAI API key (uses environment variable if None)
            debug: Whether to print debug information
        """
        self.max_workers = max_workers
        self.model = model
        self.temperature = temperature
        self.debug = debug

        # Set up OpenAI client (api_key handled by openai library via env var or explicit set)
        if api_key:
            openai.api_key = api_key
        else:
             # Use environment variable OPENAI_API_KEY
            pass # openai library handles this automatically if env var is set

    def _debug_print(self, message, data=None):
        """Print debug information if debug mode is enabled."""
        if not self.debug:
            return

        print(message)
        if data is not None:
            print(data)
            print("-" * 40)

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the GPT model for translation."""
        return f"""
        Your task is to translate an Tweet from English to German. 

        The goal is to translate the Tweet in a way that follows exactly the English tweets tone of voice. 

        Keep the tone of voice exactly as it is.
        Don't translate word by word. Translate like a native speaker would do it.
        If the tweet contains informalities, translate that as well to German.

        Again, the goal is to have the translated tweet as closely to the original english tweet as possible while still making space for changing sentence structure to match how a native speaking person would say it.

        Return only the translated tweet. Don't return anything else. Only return the translated tweet.

        If you return anything else but the translated tweet my code will break and we will loose a lot of money. So only return the translated tweet.
        """

    def _translate_single_text(self, text: str) -> Dict[str, Any]:
        """
        Translate a single piece of text using GPT.

        Args:
            text: The text to translate

        Returns:
            Dictionary with translation result and metadata
        """
        if not isinstance(text, str) or not text.strip():
             self._debug_print(f"Warning: Received empty or non-string text for translation. Skipping API call.")
             return {
                "translated_text": "[EMPTY_TEXT]",
                "success": True, # Treat empty text as a non-failure
                "elapsed_time": 0,
                "raw_response": ""
             }

        start_time = time.time()

        try:
            prompt = f"Translate the following tweet into German:\n\n{text}"

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )

            translated_text = response.choices[0].message.content.strip() # type: ignore

            elapsed_time = time.time() - start_time

            return {
                "translated_text": translated_text,
                "success": True,
                "elapsed_time": elapsed_time,
                "raw_response": response.choices[0].message.content # type: ignore
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            self._debug_print(f"Error in translation: {str(e)}")

            return {
                "translated_text": "[TRANSLATION_FAILED]",
                "success": False,
                "error": str(e),
                "elapsed_time": elapsed_time
            }

    def translate_batch_of_texts(
        self,
        texts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Translate a batch of texts in parallel.

        Args:
            texts: List of texts to translate

        Returns:
            List of translation results for each text
        """
        results = []

        def translate_text_worker(text):
            return self._translate_single_text(text)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for result in tqdm(
                executor.map(translate_text_worker, texts),
                total=len(texts),
                desc=f"Translating texts to German"
            ):
                results.append(result)

        return results

    def translate_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Translate texts in a specified DataFrame column.

        Args:
            df: DataFrame containing the text column
            text_column: Name of the column containing the text to translate

        Returns:
            DataFrame with added translation results columns ('translated_text', etc.)
        """
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame.")

        texts_to_translate = df[text_column].tolist()

        translation_results = self.translate_batch_of_texts(texts_to_translate)

        translation_results_df = pd.DataFrame(translation_results)

        # Concatenate results back to the original DataFrame
        # Ensure indexes align - reset index on both if needed
        translated_df = pd.concat([df.reset_index(drop=True), translation_results_df], axis=1)

        return translated_df