

## Dataset

There are 2 options to access the dataset.

#### Option 1: Re-create the full dataset
By running the files without having the data available will re-create the dataset from ground up. It takes about 6 hours and costs about 50 Euros in OpenAI Credits.

To do this just provide your OpenAI API key in the .env file 
```.env
OPENAI_API_KEY="sk-proj-xxxx...."
```

And continue to the next steps.

#### Option 2: Download the full dataset

Head over to this link and download the zipped dataset.

Then extract it and move it in the `data` folder: `data/datasets/...`.

Now you can continue.

## Run the Classification Benchmark

#### Classification Benchmark Goal
Figure out which models perform the best for our emotion classification task.

#### Set up Environment
- Wheter you want to run it on a GPU or on CPU is up to you. Set that environment up
- Install requirements `pip install -r requirements.txt`
- Make file executeable `sudo chmod +x run_classification_benchmark.sh`

#### Define Hyperparameters

Set your to either `cuda` if you run on the GPU, to `cpu` if you run on CPU or `mps` if you run it on Apple MX (e.g. M1, M2) chips.

You can also change the models to test as you wish. Generally no need to do this tho.
```bash
declare -a model_configs=(
    "sentence-transformers/distiluse-base-multilingual-cased-v1;false;none"
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2;false;none"
    "sentence-transformers/distiluse-base-multilingual-cased-v2;false;none"
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2;false;none"
    "intfloat/multilingual-e5-large;true;query"
    "intfloat/multilingual-e5-large-instruct;true;e5_large_instruct"
    "intfloat/multilingual-e5-base;true;query"
    "intfloat/multilingual-e5-small;true;query"
    "ibm-granite/granite-embedding-278m-multilingual;false;none"
    "ibm-granite/granite-embedding-107m-multilingual;false;none"
    "Alibaba-NLP/gte-multilingual-base;true;none"
    "mixedbread-ai/deepset-mxbai-embed-de-large-v1;false;none"
    "nomic-ai/nomic-embed-text-v2-moe;false;none"
    "Snowflake/snowflake-arctic-embed-l-v2.0;false;query"
    "shibing624/text2vec-base-multilingual;false;none"
)
```

The model definition works as follows:
- Format: "model_name_or_path;normalize_flag;prefix_function_name"
- normalize_flag: "true" or "false" (will be converted to --normalize-embeddings if true)
- prefix_function_name is one of "non", "query", "passage", "e5_large_instruct". It's basically allowing us to call a lambda function that changes the to be embedded email based on the requirements. For example if it's an instruct model, we need to put "query: " in front of the text.

More functions can be added in `benchmark_performance/service.py` and `benchmark_classification/service.py`
```python
def prefix_query(text: str) -> str: return "query: " + text
def prefix_passage(text: str) -> str: return "passage: " + text

def prefix_e5_large_instruct(text: str) -> str:
    EMOTION_MAPPING: dict[str, int] = {'Anger': 0, 'Neutral': 1, 'Joy': 2, 'Sadness': 3, 'Fear': 4} # Define this within the function or ensure it's accessible
    return f"Instruct: Classify the emotion expressed in the given Twitter message into one of the {len(EMOTION_MAPPING)} emotions: {', '.join(EMOTION_MAPPING.keys())}.\nQuery: " + text

PREFIX_FUNCTIONS: Dict[str, Optional[Callable[[str], str]]] = {
    "none": None,
    "query": prefix_query,
    "passage": prefix_passage,
    "e5_large_instruct": prefix_e5_large_instruct,
}
```


## Run Performance Benchmark

#### Performance Benchmark Goal
The goal is to figure out the following metrics of our multilingual embedding models
- Speed: How fast is the model inference?
- Memory Footprint: How big is the loaded model in memory?

#### Set up Environment
- Start server or laptop
- Install requirements `pip install -r requirements.txt`
- Make file executeable `sudo chmod +x run_performance_benchmark.sh`

#### Define Hyperparameters
The 2 parameters to change are:

- NUM_SENTENCES=100_000: How often the model should do inference on the sentences
- TARGET_DEVICE="cuda": The hardware you're using. Could also be "cpu" or "mps". Depends on the benchmark you want to run.

Other things are the embedding models. But generally not needed to change this!

#### Start Benchmark
- To start the benchmark just execute the file `./run_performance_benchmark.sh`

#### Performance Benchmark Result
- 