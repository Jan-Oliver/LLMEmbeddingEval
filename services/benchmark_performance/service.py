# benchmarking_service/benchmark_runner.py

import os
import time
import json
from typing import Callable, Optional, Dict, Any, List
import psutil
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import gc
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- RAM Measurement Utilities ---
def get_ram_usage_psutil() -> float:
    """Returns current RAM usage (RSS) of the process in MB using psutil."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

RESOURCE_AVAILABLE = False
if sys.platform != "win32":
    try:
        import resource
        RESOURCE_AVAILABLE = True
    except ImportError:
        logger.warning("Python 'resource' module not available on this platform. Will only use psutil for system RAM.")

def get_ram_usage_resource() -> Optional[float]:
    """
    Returns current max RAM usage (ru_maxrss) of the process in MB using resource module.
    Note: ru_maxrss is the peak resident set size.
    Linux returns this in kilobytes, macOS in bytes.
    """
    if not RESOURCE_AVAILABLE:
        return None
    
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # macOS returns bytes
        return usage / (1024 * 1024)
    else:  # Linux typically returns kilobytes
        return usage / 1024


def get_gpu_memory_usage_torch(device: torch.device) -> Dict[str, float]:
    """Returns current and peak GPU memory usage for the given PyTorch device in MB."""
    if not torch.cuda.is_available() or device.type != 'cuda':
        return {
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_peak_allocated_mb": 0.0, # Peak since last reset
            "gpu_memory_reserved_mb": 0.0,
            "gpu_memory_peak_reserved_mb": 0.0, # Peak since last reset
        }
    
    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    
    return {
        "gpu_memory_allocated_mb": allocated,
        "gpu_memory_peak_allocated_mb": peak_allocated,
        "gpu_memory_reserved_mb": reserved,
        "gpu_memory_peak_reserved_mb": peak_reserved,
    }

def reset_peak_gpu_memory_stats_torch(device: torch.device):
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

# Define the prefix functions
def prefix_query(text: str) -> str: return "query: " + text
def prefix_passage(text: str) -> str: return "passage: " + text

def prefix_e5_large_instruct(text: str) -> str:
    EMOTION_MAPPING: dict[str, int] = {'Anger': 0, 'Neutral': 1, 'Joy': 2, 'Sadness': 3, 'Fear': 4} # Define this within the function or ensure it's accessible
    return f"Instruct: Classify the emotion expressed in the given text message into one of the {len(EMOTION_MAPPING)} emotions: {', '.join(EMOTION_MAPPING.keys())}.\nQuery: " + text

PREFIX_FUNCTIONS: Dict[str, Optional[Callable[[str], str]]] = {
    "none": None,
    "query": prefix_query,
    "passage": prefix_passage,
    "e5_large_instruct": prefix_e5_large_instruct,
}

def run_performance_benchmark(
    model_name_or_path: str,
    use_normalize_embeddings_in_encode: bool,
    prefix_function_name: str,
    num_sentences: int,
    batch_size: int,
    target_device_str: str,
    base_results_folder: str,
    test_sentence: str,
    warmup_runs: int
) -> Dict[str, Any]:
    logger.info(f"Starting benchmark for model: {model_name_or_path} on device: {target_device_str}")
    logger.info(f"Using prefix function: {prefix_function_name}")

    benchmark_results: Dict[str, Any] = {
        "model_name": model_name_or_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": target_device_str,
        "test_sentence_used": test_sentence,
        "prefix_function_name": prefix_function_name,
        "normalize_embeddings_in_encode_call": use_normalize_embeddings_in_encode,
        "num_sentences_processed": num_sentences,
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
    }

    # --- System RAM Measurement: Initial ---
    benchmark_results["ram_psutil_before_load_mb"] = round(get_ram_usage_psutil(), 2)
    if RESOURCE_AVAILABLE:
        benchmark_results["ram_resource_maxrss_before_load_mb"] = round(get_ram_usage_resource() or 0.0, 2)
    logger.info(f"RAM (psutil) before model load: {benchmark_results['ram_psutil_before_load_mb']:.2f} MB")


    # Determine device
    if target_device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        target_device_str = "cpu"
    elif target_device_str == "mps" and not torch.backends.mps.is_available(): # Requires PyTorch 1.12+
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # more robust check
             pass # MPS is available
        else:
            logger.warning("MPS requested but not available or not supported by PyTorch version. Falling back to CPU.")
            target_device_str = "cpu"
    
    device = torch.device(target_device_str)
    benchmark_results["effective_device"] = device.type # Record what device was actually used

    # GPU Memory: Reset and measure before model load
    if device.type == 'cuda':
        reset_peak_gpu_memory_stats_torch(device) # Reset before model load
        benchmark_results["gpu_mem_before_load"] = {
            k: round(v,2) for k,v in get_gpu_memory_usage_torch(device).items()
        }
        logger.info(f"GPU Memory before model load: {benchmark_results['gpu_mem_before_load']}")


    # --- Load Model ---
    model_load_start_time = time.perf_counter()
    try:
        model = SentenceTransformer(model_name_or_path, device=target_device_str)
    except Exception as e:
        logger.error(f"Failed to load model {model_name_or_path}: {e}")
        benchmark_results["error"] = str(e)
        # Save partial results
        _save_results(benchmark_results, model_name_or_path, base_results_folder)
        return benchmark_results # Exit early
        
    model_load_time_s = time.perf_counter() - model_load_start_time
    benchmark_results["model_load_time_s"] = round(model_load_time_s, 2)
    logger.info(f"Model loaded in {benchmark_results['model_load_time_s']:.2f} seconds.")

    # --- System RAM Measurement: After Model Load ---
    benchmark_results["ram_psutil_after_load_mb"] = round(get_ram_usage_psutil(), 2)
    benchmark_results["model_ram_footprint_psutil_approx_mb"] = round(
        benchmark_results["ram_psutil_after_load_mb"] - benchmark_results["ram_psutil_before_load_mb"], 2
    )
    if RESOURCE_AVAILABLE: # ru_maxrss is a peak, so it reflects peak up to this point
        benchmark_results["ram_resource_maxrss_after_load_mb"] = round(get_ram_usage_resource() or 0.0, 2)

    logger.info(f"RAM (psutil) after model load: {benchmark_results['ram_psutil_after_load_mb']:.2f} MB")
    logger.info(f"Model RAM footprint (psutil approx.): {benchmark_results['model_ram_footprint_psutil_approx_mb']:.2f} MB")
    if RESOURCE_AVAILABLE:
        logger.info(f"RAM (resource.ru_maxrss) after model load: {benchmark_results['ram_resource_maxrss_after_load_mb']:.2f} MB")
    
    # GPU Memory: Measure after model load (captures peak during load)
    if device.type == 'cuda':
        benchmark_results["gpu_mem_after_load"] = { # This will include peak during load
            k: round(v,2) for k,v in get_gpu_memory_usage_torch(device).items()
        }
        logger.info(f"GPU Memory after model load (peak during load): {benchmark_results['gpu_mem_after_load']}")
        reset_peak_gpu_memory_stats_torch(device) # Reset again before warmup/inference

    # --- Prepare Data ---
    prefix_func_to_apply = PREFIX_FUNCTIONS.get(prefix_function_name)
    if prefix_func_to_apply:
        processed_sentence = prefix_func_to_apply(test_sentence)
    else:
        processed_sentence = test_sentence
    
    sentences_to_process = [processed_sentence] * num_sentences
    warmup_sentences_list: List[str] = [processed_sentence] * warmup_runs

    logger.info(f"Prepared {num_sentences} sentences for benchmarking (plus {warmup_runs} for warmup).")
    logger.info(f"Batch size: {batch_size}, Normalize embeddings in encode(): {use_normalize_embeddings_in_encode}")

    # --- Warm-up Phase ---
    # This is crucial for ensuring model is fully loaded/initialized, CUDA kernels compiled, etc.
    logger.info("Starting warm-up phase...")
    if warmup_runs > 0 and len(warmup_sentences_list) > 0 :
        try:
            _ = model.encode(
                warmup_sentences_list,
                batch_size=batch_size,
                show_progress_bar=False, # Usually false for warmup
                normalize_embeddings=use_normalize_embeddings_in_encode
            )
            if device.type == 'cuda': torch.cuda.synchronize() # Wait for GPU to finish
        except Exception as e:
            logger.error(f"Error during warm-up for model {model_name_or_path}: {e}")
            # Continue, but log it
    logger.info("Warm-up phase completed.")
    
    # GPU Memory: Measure after warm-up (captures peak during warm-up)
    # System RAM also captured after warm-up to see its state
    benchmark_results["ram_psutil_after_warmup_mb"] = round(get_ram_usage_psutil(), 2)
    if RESOURCE_AVAILABLE:
        benchmark_results["ram_resource_maxrss_after_warmup_mb"] = round(get_ram_usage_resource() or 0.0, 2)

    if device.type == 'cuda':
        benchmark_results["gpu_mem_after_warmup"] = { # Includes peak during warmup
            k: round(v,2) for k,v in get_gpu_memory_usage_torch(device).items()
        }
        logger.info(f"GPU Memory after warm-up (peak during warmup): {benchmark_results['gpu_mem_after_warmup']}")
        reset_peak_gpu_memory_stats_torch(device) # Reset for main inference measurement

    # --- Inference Benchmark ---
    logger.info("Starting inference benchmark...")
    inference_start_time = time.perf_counter()
    
    try:
        embeddings = model.encode(
            sentences_to_process,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=use_normalize_embeddings_in_encode
        )
        if device.type == 'cuda': torch.cuda.synchronize() # Wait for GPU to finish
        
        # Ensure embeddings are computed and potentially moved to CPU to free VRAM
        # and to reflect state if embeddings were to be used by CPU later.
        if isinstance(embeddings, np.ndarray):
            _ = embeddings.shape 
        elif isinstance(embeddings, torch.Tensor):
            _ = embeddings.shape
            if embeddings.device.type != 'cpu':
                embeddings = embeddings.cpu() # Move to CPU
                if device.type == 'cuda': torch.cuda.synchronize() 
        
        # Store a small piece of info about embeddings to ensure they are not optimized away
        benchmark_results["output_embedding_dim"] = embeddings.shape[1] if len(embeddings) > 0 and hasattr(embeddings, 'shape') else None


    except Exception as e:
        logger.error(f"Error during inference for model {model_name_or_path}: {e}")
        benchmark_results["error"] = str(e)
        inference_time_s = time.perf_counter() - inference_start_time # Time until error
        benchmark_results["inference_time_s"] = round(inference_time_s, 2)
        # Capture memory state at error
        benchmark_results["ram_psutil_at_error_mb"] = round(get_ram_usage_psutil(), 2)
        if RESOURCE_AVAILABLE: benchmark_results["ram_resource_maxrss_at_error_mb"] = round(get_ram_usage_resource() or 0.0, 2)
        if device.type == 'cuda': benchmark_results["gpu_mem_at_error"] = {k: round(v,2) for k,v in get_gpu_memory_usage_torch(device).items()}
        
        _save_results(benchmark_results, model_name_or_path, base_results_folder)
        # Clean up model from memory
        del model
        if device.type == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        return benchmark_results

    inference_time_s = time.perf_counter() - inference_start_time
    benchmark_results["inference_time_s"] = round(inference_time_s, 2)
    logger.info(f"Inference completed in {benchmark_results['inference_time_s']:.2f} seconds.")

    # --- System RAM Measurement: After Inference ---
    # This captures RAM after embeddings are potentially stored in 'embeddings' variable (now on CPU)
    benchmark_results["ram_psutil_after_inference_mb"] = round(get_ram_usage_psutil(), 2)
    if RESOURCE_AVAILABLE: # This will be the peak RSS during the whole run up to this point.
        benchmark_results["ram_resource_maxrss_after_inference_mb"] = round(get_ram_usage_resource() or 0.0, 2)
    logger.info(f"RAM (psutil) after inference: {benchmark_results['ram_psutil_after_inference_mb']:.2f} MB")
    if RESOURCE_AVAILABLE:
        logger.info(f"RAM (resource.ru_maxrss) after inference (overall peak): {benchmark_results['ram_resource_maxrss_after_inference_mb']:.2f} MB")

    # GPU Memory: Measure after inference (captures peak during inference)
    if device.type == 'cuda':
        benchmark_results["gpu_mem_after_inference"] = { # Includes peak during inference
            k: round(v,2) for k,v in get_gpu_memory_usage_torch(device).items()
        }
        logger.info(f"GPU Memory after inference (peak during inference): {benchmark_results['gpu_mem_after_inference']}")


    # --- Calculate Metrics ---
    sentences_per_second = 0
    if benchmark_results["inference_time_s"] > 0:
        sentences_per_second = num_sentences / benchmark_results["inference_time_s"]
    benchmark_results["sentences_per_second"] = round(sentences_per_second, 2)
    logger.info(f"Inference speed: {benchmark_results['sentences_per_second']:.2f} sentences/second")

    # --- Save Results ---
    _save_results(benchmark_results, model_name_or_path, base_results_folder)

    # --- Clean up ---
    logger.info("Cleaning up resources...")
    del model
    del sentences_to_process
    del embeddings
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    benchmark_results["ram_psutil_after_cleanup_mb"] = round(get_ram_usage_psutil(), 2)
    if RESOURCE_AVAILABLE: # Peak RSS should remain the same or slightly higher if cleanup allocated
        benchmark_results["ram_resource_maxrss_after_cleanup_mb"] = round(get_ram_usage_resource() or 0.0, 2)
    logger.info(f"RAM (psutil) after cleanup: {benchmark_results['ram_psutil_after_cleanup_mb']:.2f} MB")

    return benchmark_results

def _save_results(results_dict: Dict, model_name_or_path: str, base_results_folder: str):
    """Helper function to save results to a JSON file."""
    try:
        parts = model_name_or_path.replace(":", "_").split('/') # Sanitize for path
        model_specific_folder_name = os.path.join(*parts)
        model_specific_folder_path = os.path.join(base_results_folder, model_specific_folder_name)
        
        os.makedirs(model_specific_folder_path, exist_ok=True)
        
        # Sanitize device string for filename
        device_str_safe = results_dict.get("effective_device", results_dict.get("device", "unknown_device")).replace(":", "")

        results_filename = f"benchmark_{device_str_safe}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        results_filepath = os.path.join(model_specific_folder_path, results_filename)
        
        with open(results_filepath, 'w') as f:
            json.dump(results_dict, f, indent=4)
        logger.info(f"Benchmark results saved to: {results_filepath}")
    except Exception as e:
        logger.error(f"Failed to save results for {model_name_or_path}: {e}")
        results_dict["save_error"] = str(e) # Add error to dict if it's not already there from main failure

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Sentence Embedding Models")
    parser.add_argument("--model-name", required=True, help="Name or path of the Sentence Transformer model")
    parser.add_argument("--normalize-embeddings", action='store_true', help="Use normalize_embeddings=True in model.encode()")
    parser.add_argument("--prefix-function-name", default="none", choices=PREFIX_FUNCTIONS.keys(), help="Name of the prefix function to use")
    parser.add_argument("--num-sentences", type=int, default=10000, help="Total number of sentences to process")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", default="cpu", help="Device to run on (e.g., 'cpu', 'cuda', 'cuda:0', 'mps')")
    parser.add_argument("--base-results-folder", default="results_cli", help="Base folder to save results")
    parser.add_argument("--test-sentence", default="This is a moderately long test sentence for our benchmarking process.", help="The sentence to use for benchmarking")
    parser.add_argument("--warmup-runs", type=int, default=100, help="Number of sentences for warmup")

    args = parser.parse_args()

    # Create base results folder if it doesn't exist
    os.makedirs(args.base_results_folder, exist_ok=True)

    run_performance_benchmark(
        model_name_or_path=args.model_name,
        use_normalize_embeddings_in_encode=args.normalize_embeddings,
        prefix_function_name=args.prefix_function_name,
        num_sentences=args.num_sentences,
        batch_size=args.batch_size,
        target_device_str=args.device,
        base_results_folder=args.base_results_folder,
        test_sentence=args.test_sentence,
        warmup_runs=args.warmup_runs
    )