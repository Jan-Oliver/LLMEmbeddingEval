import csv
import random
from pathlib import Path
import itertools

import openai

from .views import EmailMetadata

class EmailDatasetGenerator:
    """
    Service for generating synthetic emails via OpenAI API.
    Ensures even distribution of emotions across the dataset.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        output_path: Path,
        target_size: int,
        industries: list[str],
        emotions: list[str],
    ) -> None:
        """
        Initialize the EmailDatasetGenerator.
        
        Args:
            api_key: The API key for the OpenAI API.
            model: The model to use for generating the emails.
            output_path: The path to save the generated emails.
            target_size: The number of emails to generate.
            industries: The industries to generate emails for.
            emotions: The emotions to generate emails for.
        """
        openai.api_key = api_key
        self.model = model
        self.industries = industries
        self.emotions = emotions
        self.target_size = target_size
        self.output_path = output_path

        # Calculate how many samples per emotion to ensure even distribution
        self.samples_per_emotion = self.target_size // len(self.emotions)
        
        # Create balanced distribution of emotions
        self.emotion_distribution = []
        for emotion in self.emotions:
            self.emotion_distribution.extend([emotion] * self.samples_per_emotion)
        
        # Handle any remainder to reach the target size
        remainder = self.target_size - len(self.emotion_distribution)
        if remainder > 0:
            additional_emotions = self.emotions[:remainder]
            self.emotion_distribution.extend(additional_emotions)
            
        # Shuffle the emotions to avoid having all samples of the same emotion consecutively
        random.shuffle(self.emotion_distribution)

    def _build_prompt(self, meta: EmailMetadata) -> str:
        return f""" 
        Here are the metadata parameters to use:

        - Industry: {meta.industry}
        - Emotion: {meta.emotion}
        """

    def _generate(self, meta: EmailMetadata) -> str:
        system_prompt = """
        Given your knowledge of the ENRON dataset, create an email that could be part of that dataset but changed based on provided hyperparameters.

        The structure of the created email has to follow the ENRON dataset email as closely as possible.

        Only change the content of the reference email to match the provided hyperparameters. But still try to keep it as closely related to the original email as possible.

        The provided Hyperparameters are:
        - Industry: Your task is to adapt the reference email from the ENRON dataset to sound as if it was send by somebody else from <Industry>. <Industry> can be for example Finance, Biotech, or anything else. This might mean you need to change the purpose of the email. But make sure to keep it as close as possible to the actual email.
        - Emotion: Your task is to adapt the reference email from the ENRON dataset to match the emotion. This might mean you have to make the email more joyful or sad. You are allowed to change the reference email but try to follow the structure as closely as possible.
        """

        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._build_prompt(meta)}
            ],
            temperature=0.7
        )
        return resp.choices[0].message.content.strip() # type: ignore

    def _write_header(self) -> None:
        if not self.output_path.exists():
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[*EmailMetadata.schema()["properties"].keys(), "email_text"],
                    quoting=csv.QUOTE_ALL,
                    escapechar='\\'
                )
                writer.writeheader()

    def _append(self, meta: EmailMetadata, text: str) -> None:
        row = meta.dict()
        row["email_text"] = text
        with open(self.output_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=row.keys(),
                quoting=csv.QUOTE_ALL,
                escapechar='\\'
            )
            writer.writerow(row)

    def run(self) -> None:
        self._write_header()
        for idx, emotion in enumerate(self.emotion_distribution, start=1):
            # Randomly select an industry for each email
            industry = random.choice(self.industries)
            
            meta = EmailMetadata(
                industry=industry,
                emotion=emotion
            )
            
            try:
                text = self._generate(meta)
            except Exception as e:
                print(f"Error generating sample {idx}: {e}")
                continue
                
            self._append(meta, text)
            print(f"{idx}/{self.target_size} done")
            
    def get_distribution_stats(self) -> dict:
        """
        Returns statistics about the distribution of emotions in the dataset.
        Useful for verification that the distribution is even.
        """
        stats = {}
        for emotion in self.emotions:
            count = self.emotion_distribution.count(emotion)
            stats[emotion] = {
                "count": count,
                "percentage": (count / self.target_size) * 100
            }
        return stats