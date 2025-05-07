import pandas as pd
import os
import time
import json
import concurrent.futures
from tqdm import tqdm
from collections import Counter
import openai
from typing import List, Dict, Any, Optional

class EnronEmailClassifier:
    """
    Classify emotions in Enron emails using GPT, with multiple classifications
    per email and parallel processing.
    """
    
    def __init__(
        self,
        target_emotions: List[str],
        num_classifications: int = 5,
        max_workers: int = 5,
        model: str = "gpt-4.1",
        temperature: float = 1.0,
        api_key: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize the email classifier.
        
        Args:
            target_emotions: List of emotion categories to classify
            num_classifications: Number of times to classify each email
            max_workers: Maximum number of concurrent workers for API calls
            model: GPT model to use
            temperature: Temperature for the GPT model (0.0 to 2.0)
            api_key: OpenAI API key (uses environment variable if None)
            debug: Whether to print debug information
        """
        self.target_emotions = target_emotions
        self.num_classifications = num_classifications
        self.max_workers = max_workers
        self.model = model
        self.temperature = temperature
        self.debug = debug
        
        # Set up OpenAI client
        if api_key:
            openai.api_key = api_key
        else:
            # Use environment variable OPENAI_API_KEY
            pass
    
    def _debug_print(self, message, data=None):
        """Print debug information if debug mode is enabled."""
        if not self.debug:
            return
            
        print(message)
        if data is not None:
            print(data)
            print("-" * 40)
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the GPT model."""
        # Join target emotions with formatting
        emotion_list = "\n".join([f"* {emotion}" for emotion in self.target_emotions])
        
        return f"""
        You are an expert emotion analyst. Your task is to classify the emotion expressed in each email into one of the following categories:
        {emotion_list}
        
        For each email, analyze the content and determine the most prominent emotion.
        Respond with ONLY a valid JSON object with a key 'emotion' whose value is one of: {', '.join([f"'{emotion}'" for emotion in self.target_emotions])}.
        Do not include any explanation or additional text.
        """
    
    def _classify_single_email(self, email_text: str, attempt_id: int) -> Dict[str, Any]:
        """
        Classify a single email using GPT.
        
        Args:
            email_text: The preprocessed email text to classify
            attempt_id: The ID of this classification attempt
            
        Returns:
            Dictionary with classification result and metadata
        """
        start_time = time.time()
        
        try:
            # Prepare the user prompt
            prompt = f"Classify the following email into one emotion category. Reply with a valid JSON object with a key 'emotion' whose value is one of: {', '.join([f'{emotion}' for emotion in self.target_emotions])}.\n\n"
            prompt += f"Email:\n{email_text}\n\n"
            
            # Make API call to OpenAI
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            response_content = response.choices[0].message.content.strip() # type: ignore
            emotion_data = json.loads(response_content)
            
            # Extract the emotion from the response
            if "emotion" in emotion_data:
                emotion = emotion_data["emotion"]
                if emotion not in self.target_emotions:
                    self._debug_print(f"Warning: Received invalid emotion '{emotion}'. Using fallback.")
                    emotion = self.target_emotions[0]  # Use first emotion as fallback
            else:
                # Fallback if the response doesn't have an "emotion" key
                self._debug_print(f"Warning: Response doesn't have an 'emotion' key. Using fallback.")
                self._debug_print(f"Response: {emotion_data}")
                emotion = self.target_emotions[0]  # Use first emotion as fallback
            
            elapsed_time = time.time() - start_time
            
            return {
                "emotion": emotion,
                "attempt_id": attempt_id,
                "success": True,
                "elapsed_time": elapsed_time,
                "raw_response": response_content
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self._debug_print(f"Error in classification attempt {attempt_id}: {str(e)}")
            
            return {
                "emotion": self.target_emotions[0],  # Use first emotion as fallback
                "attempt_id": attempt_id,
                "success": False,
                "error": str(e),
                "elapsed_time": elapsed_time
            }
    
    def _classify_email_multiple_times(self, email_id: int, email_text: str) -> Dict[str, Any]:
        """
        Classify a single email multiple times and aggregate results.
        
        Args:
            email_id: The ID of the email in the dataset
            email_text: The preprocessed email text to classify
            
        Returns:
            Dictionary with all classification attempts and aggregated result
        """
        # Run multiple classifications
        results = []
        
        for attempt in range(self.num_classifications):
            result = self._classify_single_email(email_text, attempt)
            results.append(result)
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
        
        # Count occurrences of each emotion
        emotions = [r["emotion"] for r in results]
        emotion_counts = Counter(emotions)
        
        # Find the most common emotion (majority vote)
        majority_emotion, majority_count = emotion_counts.most_common(1)[0]
        
        return {
            "email_id": email_id,
            "final_emotion": majority_emotion,
            "confidence": majority_count / self.num_classifications,
            "classification_attempts": results,
            "emotion_counts": dict(emotion_counts)
        }
    
    def classify_batch_of_emails(
        self, 
        email_texts: List[str], 
        start_index: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Classify a batch of emails in parallel.
        
        Args:
            email_texts: List of preprocessed email texts to classify
            start_index: Starting index for email_id (useful for batches)
            
        Returns:
            List of classification results for each email
        """
        results = []
        
        # Define a helper function for the worker
        def classify_email_worker(args):
            idx, email = args
            return self._classify_email_multiple_times(idx, email)
        
        # Process emails in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a list of (email_id, email_text) tuples
            email_tasks = [(start_index + i, email) for i, email in enumerate(email_texts)]
            
            # Map the tasks to the worker function with a progress bar
            for result in tqdm(
                executor.map(classify_email_worker, email_tasks),
                total=len(email_tasks),
                desc="Classifying emails"
            ):
                results.append(result)
        
        return results
    
    def classify_emails(self, emails_df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Classify all emails in a DataFrame.
        
        Args:
            emails_df: DataFrame containing emails
            text_column: Name of the column containing the email text
            
        Returns:
            DataFrame with added emotion columns
        """
        # Extract email texts
        email_texts = emails_df[text_column].tolist()
        
        # Classify all emails
        classification_results = self.classify_batch_of_emails(email_texts)
        
        # Create a new DataFrame with all the detailed results
        result_df = pd.DataFrame(classification_results)
        
        # Extract classification attempts as separate columns
        for attempt_id in range(self.num_classifications):
            result_df[f'emotion_attempt_{attempt_id+1}'] = result_df['classification_attempts'].apply(
                lambda attempts: next((a['emotion'] for a in attempts if a['attempt_id'] == attempt_id), None)
            )
        
        # Merge with the original DataFrame
        result_df.drop(columns=['classification_attempts'], inplace=True)
        final_df = pd.concat([emails_df.reset_index(drop=True), result_df], axis=1)
        
        return final_df
    
    def save_classification_results(self, df: pd.DataFrame, output_path: str) -> None:
        """Save classification results to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Classification results saved to {output_path}")