# monitoring/service.py
import time
import psutil
import os
import json
import uuid # For unique event IDs
from datetime import datetime, timezone # Ensure timezone aware datetimes
from typing import List, Literal, Optional, Union, Dict, Any
from pathlib import Path

# Import Pydantic models
from .views import ModelLoadEventData, InferenceRunEventData, BaseEventData, MonitoringEventData

class MonitoringService:
    """
    A service to monitor and record performance metrics for various events
    including model loading and inference runs, using Pydantic for typesafe data.
    """
    def __init__(self, context_name: str, output_dir: Path):
        self.context_name = context_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.recorded_events: List[MonitoringEventData] = []
        
        # Temporary storage for data before Pydantic model creation
        self._current_event_construction_data: Dict[str, Any] = {}
        
        try:
            self._process = psutil.Process(os.getpid())
        except psutil.Error:
            print(f"Warning: Could not get current process details for psutil for context '{self.context_name}'. RAM monitoring might be affected.")
            self._process = None

        self._event_start_time_ns: Optional[int] = None
        self._event_start_ram_mb: Optional[float] = None
        self._current_event_type: Optional[Literal["model_load", "inference_run"]] = None


    def _get_current_ram_mb(self) -> Optional[float]:
        if self._process:
            try:
                return self._process.memory_info().vms / (1024 * 1024)
            except psutil.Error as e:
                print(f"Warning: psutil error when getting RAM for context '{self.context_name}': {e}")
                return None
        return None

    def _start_event_tracking(self, event_type: Literal["model_load", "inference_run"], event_specific_data: Optional[Dict[str, Any]] = None) -> None:
        if self._event_start_time_ns is not None:
            print(f"Warning: Monitoring for a previous event was not stopped before starting new event '{event_type}'. Overwriting.")

        self._event_start_time_ns = time.perf_counter_ns()
        self._event_start_ram_mb = self._get_current_ram_mb()
        self._current_event_type = event_type

        self._current_event_construction_data = {
            "event_id": uuid.uuid4(),
            "context_name": self.context_name,
            "event_type": event_type, # This will be validated by Pydantic model
            "timestamp_start_iso": datetime.now(timezone.utc),
            "ram_usage_start_mb": self._event_start_ram_mb,
        }
        if event_specific_data:
            self._current_event_construction_data.update(event_specific_data)

    def _stop_event_tracking(self) -> MonitoringEventData:
        if self._event_start_time_ns is None or self._current_event_type is None:
            raise RuntimeError(f"Monitoring for context '{self.context_name}' was not started or event type is missing. Call a start_*_monitoring() method first.")

        event_end_time_ns = time.perf_counter_ns()
        event_end_ram_mb = self._get_current_ram_mb()
        duration_seconds = (event_end_time_ns - self._event_start_time_ns) / 1_000_000_000.0

        self._current_event_construction_data.update({
            "timestamp_end_iso": datetime.now(timezone.utc),
            "ram_usage_end_mb": event_end_ram_mb,
            "duration_seconds": duration_seconds,
        })

        if self._event_start_ram_mb is not None and event_end_ram_mb is not None:
            self._current_event_construction_data["ram_usage_delta_mb"] = event_end_ram_mb - self._event_start_ram_mb
        
        # Create Pydantic model instance
        event_data_model: MonitoringEventData
        if self._current_event_type == "model_load":
            event_data_model = ModelLoadEventData(**self._current_event_construction_data)
        elif self._current_event_type == "inference_run":
            # Add inference-specific calculations before creating the model
            num_texts = self._current_event_construction_data.get("num_texts_processed")
            total_chars = self._current_event_construction_data.get("total_chars_processed")
            
            if num_texts and num_texts > 0 and duration_seconds > 0:
                self._current_event_construction_data["seconds_per_text"] = duration_seconds / num_texts
            
            if total_chars and total_chars > 0 and duration_seconds > 0:
                self._current_event_construction_data["seconds_per_1k_chars"] = (duration_seconds / total_chars) * 1000
            
            event_data_model = InferenceRunEventData(**self._current_event_construction_data)
        else:
            # Should not happen if logic is correct
            raise ValueError(f"Unknown current event type: {self._current_event_type}")

        self.recorded_events.append(event_data_model)

        # Reset for the next potential event
        self._current_event_construction_data = {}
        self._event_start_time_ns = None
        self._event_start_ram_mb = None
        self._current_event_type = None
        
        return event_data_model

    # --- Model Load Monitoring ---
    def start_model_load_monitoring(self) -> None:
        self._start_event_tracking(event_type="model_load")

    def stop_model_load_monitoring(self) -> ModelLoadEventData:
        event_data = self._stop_event_tracking()
        if not isinstance(event_data, ModelLoadEventData):
            raise TypeError("Expected ModelLoadEventData, got different type.") # Should be guaranteed by logic
        return event_data

    # --- Inference Run Monitoring ---
    def start_inference_monitoring(self, num_texts: Optional[int] = None, total_chars: Optional[int] = None) -> None:
        event_specific_data = {}
        if num_texts is not None:
            event_specific_data["num_texts_processed"] = num_texts
        if total_chars is not None:
            event_specific_data["total_chars_processed"] = total_chars
            if num_texts and num_texts > 0:
                 event_specific_data["avg_chars_per_text"] = float(total_chars) / num_texts
        
        self._start_event_tracking(event_type="inference_run", event_specific_data=event_specific_data)

    def stop_inference_monitoring(self) -> InferenceRunEventData:
        event_data = self._stop_event_tracking()
        if not isinstance(event_data, InferenceRunEventData):
            raise TypeError("Expected InferenceRunEventData, got different type.") # Should be guaranteed by logic
        return event_data

    def get_all_events(self) -> List[MonitoringEventData]:
        return self.recorded_events

    def save_all_events(self, filename: Optional[str] = None) -> str:
        if not self.recorded_events:
            print(f"No events recorded for context '{self.context_name}' to save.")
            return ""

        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_context_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in self.context_name)
            filename = f"monitoring_events_{safe_context_name}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        try:
            # Use Pydantic's model_dump for proper serialization
            events_to_save = [event.model_dump(mode="json") for event in self.recorded_events]
            with open(filepath, 'w') as f:
                json.dump(events_to_save, f, indent=4)
            print(f"Monitoring events for '{self.context_name}' saved to {filepath}")
            return str(filepath)
        except IOError as e:
            print(f"Error saving metrics to {filepath}: {e}")
            return ""
        except Exception as e: # Catch other potential errors during model_dump
            print(f"An unexpected error occurred during event serialization: {e}")
            return ""


    def analyze_pc_compatibility(self, metrics_source: Union[str, List[MonitoringEventData], None] = None) -> None:
        print("\n--- PC Compatibility Analysis (Conceptual - Focus on Inference) ---")
        
        inference_events: List[InferenceRunEventData] = []
        source_description = ""

        if isinstance(metrics_source, str): # File path
            try:
                with open(metrics_source, 'r') as f:
                    all_event_dicts = json.load(f)
                # Parse dicts into Pydantic models, filtering for inference runs
                for event_dict in all_event_dicts:
                    if event_dict.get("event_type") == "inference_run":
                        try:
                            inference_events.append(InferenceRunEventData(**event_dict))
                        except Exception as e: # Pydantic validation error
                             print(f"Skipping event during load due to parsing error: {event_dict.get('event_id')}, Error: {e}")
                source_description = f"Loaded {len(inference_events)} inference events from {metrics_source}"
            except (IOError, json.JSONDecodeError) as e:
                print(f"Could not load or parse metrics from {metrics_source}: {e}")
                return
        elif isinstance(metrics_source, list): # In-memory list of Pydantic objects
            inference_events = [event for event in metrics_source if isinstance(event, InferenceRunEventData)]
            source_description = f"Analyzing {len(inference_events)} in-memory inference events"
        elif metrics_source is None and self.recorded_events: # Use service's current events
            inference_events = [event for event in self.recorded_events if isinstance(event, InferenceRunEventData)]
            source_description = f"Analyzing {len(inference_events)} current in-memory inference events"
        else:
            print("No metrics source provided or available in the service.")
            return

        if not inference_events:
            print(f"{source_description}, but no inference events found to analyze for PC compatibility.")
            return
        print(source_description + ".")

        pc_profiles = {
            "Low-End Laptop (e.g., 4GB RAM, older CPU)": {
                "max_inference_ram_delta_mb": 300.0,
                "max_time_per_email_s": 1.0 
            },
            "Mid-Range Laptop (e.g., 8GB RAM, modern CPU)": {
                "max_inference_ram_delta_mb": 1024.0,
                "max_time_per_email_s": 0.5
            },
            "High-End PC (e.g., 16GB+ RAM, powerful CPU)": {
                "max_inference_ram_delta_mb": 4096.0,
                "max_time_per_email_s": 0.1
            }
        }

        for event_data in inference_events:
            # Now using Pydantic model attributes directly
            print(f"\nAnalysis for Inference Event ID {event_data.event_id} (Context: {event_data.context_name}):")
            if event_data.ram_usage_delta_mb is not None:
                print(f"  - RAM increase during inference: {event_data.ram_usage_delta_mb:.2f} MB")
            if event_data.seconds_per_text is not None:
                 print(f"  - Avg. time per text/email: {event_data.seconds_per_text:.4f} seconds")
            
            if event_data.ram_usage_delta_mb is None or event_data.seconds_per_text is None:
                print("  - Insufficient data for detailed compatibility check (missing RAM delta or time per text).")
                continue

            for profile_name, specs in pc_profiles.items():
                # Ensure comparison with float values
                ram_delta = float(event_data.ram_usage_delta_mb)
                time_per_email = float(event_data.seconds_per_text)

                ram_ok = ram_delta <= specs["max_inference_ram_delta_mb"]
                time_ok = time_per_email <= specs["max_time_per_email_s"]
                
                compatibility_notes = []
                if ram_ok and time_ok:
                    compatibility_notes.append(f"Likely compatible with: {profile_name}")
                else:
                    if not ram_ok:
                        compatibility_notes.append(f"May struggle on {profile_name} due to RAM usage (needs {ram_delta:.2f}MB delta, profile target <{specs['max_inference_ram_delta_mb']:.2f}MB).")
                    if not time_ok:
                        compatibility_notes.append(f"May be too slow for {profile_name} (takes {time_per_email:.4f}s/email, profile target <{specs['max_time_per_email_s']:.2f}s).")
                for note in compatibility_notes:
                    print(f"  - {note}")
        print("--- End of Conceptual PC Compatibility Analysis ---")