# Azure Reasoning Trace Pipeline

A simple pipeline for generating and printing reasoning traces for EHR prediction tasks using Azure OpenAI (GPT-4o).

## Overview

This pipeline generates detailed reasoning traces for medical prediction tasks in a single streamlined workflow:

1. **Data Preparation**: Automatically extracts and serializes patient data from the FEMR database
2. **Reasoning Generation**: Generates reasoning traces using Azure OpenAI (GPT-4o)

The pipeline uses a base prompt strategy without corrections, saving results to JSON files in the `reasoning_output/` directory.




## Setup

**First, configure your Azure OpenAI credentials securely:**

See [SETUP.md](SETUP.md) for detailed instructions on setting up your Azure OpenAI API key using environment variables or configuration files.


**Then, see [../../README.md](../../README.md)** to setup the EHRSHOT_ENV Conda environment. 

### 🚀 **Example Reasoning Generation**
```bash
cd /home/lrshi/llm/ehrshot-benchmark/ehrshot/reasoning

# Complete pipeline: data preparation + reasoning generation in one command
python azure_reasoning_complete.py \
    --path_to_database /home/lrshi/llm/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir /home/lrshi/llm/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark \
    --task_to_instructions /home/lrshi/llm/ehrshot-benchmark/ehrshot/serialization/task_to_instructions.json \
    --num_samples 3 \
    --max_examples 5 \
    --output_file /home/lrshi/llm/ehrshot-benchmark/ehrshot/reasoning_output/azure_reasoning_traces.json \
    --temperature 0.7 \
    --max_tokens 4096
```