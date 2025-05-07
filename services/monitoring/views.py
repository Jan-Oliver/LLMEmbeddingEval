# monitoring/views.py
from typing import Optional, Literal
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

class BaseEventData(BaseModel):
    """
    Base model for all monitoring events.
    """
    event_id: UUID = Field(..., description="Unique identifier for the event.")
    context_name: str = Field(..., description="Context name (e.g., model name, pipeline stage).")
    event_type: str # This will be more specific in derived classes using Literal
    
    timestamp_start_iso: datetime = Field(..., description="Start timestamp of the event in ISO format.")
    timestamp_end_iso: datetime = Field(..., description="End timestamp of the event in ISO format.")
    duration_seconds: float = Field(..., description="Duration of the event in seconds.")
    
    ram_usage_start_mb: Optional[float] = Field(None, description="RAM usage in MB at the start of the event.")
    ram_usage_end_mb: Optional[float] = Field(None, description="RAM usage in MB at the end of the event.")
    ram_usage_delta_mb: Optional[float] = Field(None, description="Change in RAM usage in MB during the event.")

class ModelLoadEventData(BaseEventData):
    """
    Data model for model loading events.
    """
    event_type: Literal["model_load"] = "model_load"

class InferenceRunEventData(BaseEventData):
    """
    Data model for inference run events.
    """
    event_type: Literal["inference_run"] = "inference_run"
    
    num_texts_processed: Optional[int] = Field(None, description="Number of texts processed in this run.")
    total_chars_processed: Optional[int] = Field(None, description="Total characters processed in this run.")
    avg_chars_per_text: Optional[float] = Field(None, description="Average characters per text in this run.")
    seconds_per_text: Optional[float] = Field(None, description="Average processing time per text in seconds.")
    seconds_per_1k_chars: Optional[float] = Field(None, description="Processing time per 1000 characters in seconds.")

# Union type for type hinting if needed, though Pydantic handles it well with discriminated unions based on 'event_type'
MonitoringEventData = ModelLoadEventData | InferenceRunEventData