import openai
import time
import concurrent.futures
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Any, Optional

class EmailTranslator:
    """
    Translates email text using GPT.
    """

    def __init__(
        self,
        max_workers: int = 5,
        model: str = "gpt-4.1-mini", # Using the same model as classifier for consistency
        temperature: float = 0.5, # Lower temperature is often better for translation
        api_key: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize the email translator.

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
        Your task is to translate an Email from English to German. 

        The goal is to translate the Email in a way that follows exactly the English emails tone of voice. 

        Keep the tone of voice exactly as it is.
        Don't translate word by word. Translate like a native speaker would do it.
        If the email contains informalities, translate that as well to German.

        Again, the goal is to have the translated email as closely to the original english email as possible while still making space for changing sentence structure to match how a native speaking person would say it.

        Return only the translated email. Don't return anything else. Only return the translated email.

        If you return anything else but the translated email my code will break and we will loose a lot of money. So only return the translated email.
        """

    def _translate_single_email(self, email_text: str, attempt_id: int = 0) -> Dict[str, Any]:
        """
        Translate a single email using GPT.

        Args:
            email_text: The email text to translate
            attempt_id: Dummy ID for potential future retry logic

        Returns:
            Dictionary with translation result and metadata
        """
        start_time = time.time()

        try:
            prompt = f"Translate the following email to German:\n\n{email_text}"

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
            self._debug_print(f"Error in translation attempt {attempt_id}: {str(e)}")

            return {
                "translated_text": "[TRANSLATION_FAILED]", # Placeholder for failed translations
                "success": False,
                "error": str(e),
                "elapsed_time": elapsed_time
            }

    def translate_batch_of_emails(
        self,
        email_texts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Translate a batch of emails in parallel.

        Args:
            email_texts: List of email texts to translate

        Returns:
            List of translation results for each email
        """
        results = []

        # Define a helper function for the worker
        def translate_email_worker(email):
            # We don't need email_id here, just the text
            return self._translate_single_email(email)

        # Process emails in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Use map with a progress bar
            for result in tqdm(
                executor.map(translate_email_worker, email_texts),
                total=len(email_texts),
                desc=f"Translating emails to German"
            ):
                results.append(result)

        return results

    def translate_emails_df(self, emails_df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Translate emails in a DataFrame.

        Args:
            emails_df: DataFrame containing emails
            text_column: Name of the column containing the text to translate

        Returns:
            DataFrame with added translation results
        """
        # Extract email texts from the specified column
        email_texts = emails_df[text_column].tolist()

        # Translate the emails in batches
        translation_results = self.translate_batch_of_emails(email_texts)

        # Create a new DataFrame from the translation results
        translation_results_df = pd.DataFrame(translation_results)

        # Concatenate the original DataFrame with the translation results
        # We use axis=1 to add the new columns
        translated_df = pd.concat([emails_df.reset_index(drop=True), translation_results_df], axis=1)

        return translated_df