import os
import json
import pandas as pd

def aggregate_experiment_results():
    """
    Aggregates experiment results from the specified folder structure.
    """
    results = {} # Use a dictionary to store results, keyed by model_name for easier merging

    # --- Helper function to initialize a model's entry in the results dictionary ---
    def get_or_initialize_model_data(model_name):
        if model_name not in results:
            results[model_name] = {
                "Model Name": model_name,
                "de - trainings acc": None,
                "en - trainings acc": None,
                "de/en - trainings acc": None,
                "de - val acc": None,
                "en - val acc": None,
                "de/en - val acc": None,
                "de - test acc": None,
                "en - test acc": None,
                "de/en - test acc": None,
                "Memory Usage (MB)": None,
                "Embedding Dimension": None,
                "Encoding Speed (Sentences/Sec) GPU": None,
            }
        return results[model_name]

    # --- 1. Process Performance Folder ---
    performance_base_path = "benchmarking_results/performance"
    if not os.path.exists(performance_base_path):
        print(f"Warning: Performance directory not found: {performance_base_path}")
    else:
        for device_type in os.listdir(performance_base_path): # gpu, cpu
            device_path = os.path.join(performance_base_path, device_type)
            if not os.path.isdir(device_path):
                continue

            for timestamp_folder in os.listdir(device_path):
                timestamp_path = os.path.join(device_path, timestamp_folder)
                if not os.path.isdir(timestamp_path):
                    continue

                for model_family_folder in os.listdir(timestamp_path):
                    model_family_path = os.path.join(timestamp_path, model_family_folder)
                    if not os.path.isdir(model_family_path):
                        continue

                    for model_name_folder in os.listdir(model_family_path):
                        model_path = os.path.join(model_family_path, model_name_folder)
                        if not os.path.isdir(model_path):
                            continue
                        
                        json_file_path = None
                        for file_in_model_path in os.listdir(model_path):
                            if file_in_model_path.endswith(".json"):
                                json_file_path = os.path.join(model_path, file_in_model_path)
                                break 

                        if not json_file_path:
                            print(f"Warning: No JSON file found in {model_path}")
                            continue
                        
                        try:
                            with open(json_file_path, 'r') as f:
                                data = json.load(f)

                            model_name = data.get("model_name")
                            if not model_name:
                                print(f"Warning: 'model_name' not found in {json_file_path}. Skipping.")
                                continue

                            model_data_entry = get_or_initialize_model_data(model_name)

                            model_data_entry["Embedding Dimension"] = data.get("output_embedding_dim")
                            
                            if "gpu" in device_type.lower():
                                model_data_entry["Encoding Speed (Sentences/Sec) GPU"] = data.get("sentences_per_second")
                                gpu_mem_after_load = data.get("gpu_mem_after_load")
                                if gpu_mem_after_load and isinstance(gpu_mem_after_load, dict):
                                    model_data_entry["Memory Usage (MB)"] = gpu_mem_after_load.get("gpu_memory_allocated_mb")
                            elif "cpu" in device_type.lower():
                                model_data_entry[f"Encoding Speed (Sentences/Sec) {device_type}"] = data.get("sentences_per_second")

                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from {json_file_path}")
                        except Exception as e:
                            print(f"Error processing performance file {json_file_path}: {e}")


    # --- 2. Process Classification Folder ---
    classification_base_path = "benchmarking_results/classification"
    if not os.path.exists(classification_base_path):
        print(f"Warning: Classification directory not found: {classification_base_path}")
    else:
        for dataset_type_folder in os.listdir(classification_base_path): 
            dataset_path = os.path.join(classification_base_path, dataset_type_folder)
            if not os.path.isdir(dataset_path):
                continue

            for model_family_folder in os.listdir(dataset_path):
                model_family_path = os.path.join(dataset_path, model_family_folder)
                if not os.path.isdir(model_family_path):
                    continue

                for model_name_folder in os.listdir(model_family_path):
                    model_specific_path = os.path.join(model_family_path, model_name_folder)
                    if not os.path.isdir(model_specific_path):
                        continue

                    model_info_path = os.path.join(model_specific_path, "model_info.json")
                    training_metrics_path = os.path.join(model_specific_path, "training_metrics.json")
                    test_eval_path = os.path.join(model_specific_path, "test", "evaluation_results.json")


                    if not os.path.exists(model_info_path):
                        print(f"Warning: model_info.json not found in {model_specific_path}")
                        continue
                    if not os.path.exists(training_metrics_path):
                        print(f"Warning: training_metrics.json not found in {model_specific_path}")
                        continue
                    if not os.path.exists(test_eval_path):
                        print(f"Warning: evaluation_results.json not found in {os.path.join(model_specific_path, 'train')}")
                        continue

                    try:
                        with open(model_info_path, 'r') as f:
                            model_info = json.load(f)
                        
                        model_name = model_info.get("embedding_model_name")
                        if not model_name:
                            print(f"Warning: 'embedding_model_name' not found in {model_info_path}. Skipping.")
                            continue
                        
                        model_data_entry = get_or_initialize_model_data(model_name)

                        if model_data_entry["Embedding Dimension"] is None:
                             model_data_entry["Embedding Dimension"] = model_info.get("embedding_dimension")

                        epoch = model_info.get("epoch") 
                        if epoch is None:
                            print(f"Warning: 'epoch' not found in {model_info_path}. Skipping.")
                            continue
                        
                        epoch_idx = epoch

                        with open(training_metrics_path, 'r') as f:
                            training_metrics = json.load(f)
                        
                        train_acc_list = training_metrics.get("train_acc")
                        val_acc_list = training_metrics.get("val_acc")

                        current_train_acc = None
                        current_val_acc = None

                        if train_acc_list and len(train_acc_list) > epoch_idx >= 0: # ensure epoch_idx is valid
                            current_train_acc = train_acc_list[epoch_idx]
                        else:
                            print(f"Warning: train_acc for epoch {epoch} (index {epoch_idx}) not found or index out of bounds in {training_metrics_path}")
                        
                        if val_acc_list and len(val_acc_list) > epoch_idx >= 0: # ensure epoch_idx is valid
                            current_val_acc = val_acc_list[epoch_idx]
                        else:
                            print(f"Warning: val_acc for epoch {epoch} (index {epoch_idx}) not found or index out of bounds in {training_metrics_path}")

                        with open(test_eval_path, 'r') as f:
                            test_eval_metrics = json.load(f)
                        current_test_acc = test_eval_metrics.get("accuracy")

                        if dataset_type_folder == "balanced-de":
                            model_data_entry["de - trainings acc"] = current_train_acc
                            model_data_entry["de - val acc"] = current_val_acc
                            model_data_entry["de - test acc"] = current_test_acc
                        elif dataset_type_folder == "balanced-en":
                            model_data_entry["en - trainings acc"] = current_train_acc
                            model_data_entry["en - val acc"] = current_val_acc
                            model_data_entry["en - test acc"] = current_test_acc
                        elif dataset_type_folder == "balanced": 
                            model_data_entry["de/en - trainings acc"] = current_train_acc
                            model_data_entry["de/en - val acc"] = current_val_acc
                            model_data_entry["de/en - test acc"] = current_test_acc
                        elif dataset_type_folder == "unbalanced":
                            model_data_entry["de/en - trainings acc (unbalanced)"] = current_train_acc
                            model_data_entry["de/en - val acc (unbalanced)"] = current_val_acc
                            model_data_entry["de/en - test acc (unbalanced)"] = current_test_acc

                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode JSON from one of the files in {model_specific_path}: {e}")
                    except IndexError:
                        print(f"Warning: Epoch index out of bounds for {model_name} in {model_specific_path}. Epoch: {epoch}")
                    except Exception as e:
                        print(f"Error processing classification files for {model_specific_path}: {e}")
    
    if not results:
        print("No results were processed.")
        return pd.DataFrame()

    df = pd.DataFrame(list(results.values()))

    return df

if __name__ == "__main__":
    aggregated_df = aggregate_experiment_results()
    
    print("\nAggregated Results:")
    # pd.set_option('display.max_columns', None) # Show all columns
    # pd.set_option('display.width', 1000) # Adjust width for wider display
    print(aggregated_df.to_string())

    # You can save the DataFrame to a CSV file like this:
    aggregated_df.to_csv("./benchmarking_results/master_thesis_aggregated_results.csv", index=False)
    print("\nResults saved to master_thesis_aggregated_results.csv")
    
    # Also store the markdown table
    markdown_table = aggregated_df.to_markdown(index=False)
    with open("./benchmarking_results/master_thesis_aggregated_results.md", "w") as f:
        f.write(markdown_table)
    print("\nResults saved to master_thesis_aggregated_results.md")