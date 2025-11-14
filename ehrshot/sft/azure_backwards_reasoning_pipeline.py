#!/usr/bin/env python3
"""
Azure-based backwards reasoning pipeline for supervised fine-tuning.
Uses ground truth answers to generate reasoning traces that explain the correct prediction
without directly referencing the ground truth.
"""

import os
import json
import argparse
from typing import List, Dict, Optional, Any
from openai import AzureOpenAI
from loguru import logger


class AzureBackwardsReasoningConfig:
    """Configuration for Azure-based backwards reasoning trace generation"""
    
    def __init__(self,
                 endpoint: str = None,
                 api_key: str = None,
                 api_version: str = "2024-12-01-preview",
                 deployment: str = "gpt-4.1",
                 model: str = "gpt-4.1",
                 max_tokens: int = 4096,
                 temperature: float = 0.3,  # Lower temperature for more consistent reasoning
                 top_p: float = 1.0):
        
        # Try to load from config file first, then fall back to environment variables
        config_file = os.path.join(os.path.dirname(__file__), "azure_config.json")
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.endpoint = endpoint or config.get("endpoint", "https://clinicalml-cloudbank-openai.openai.azure.com/")
            self.api_key = api_key or config.get("api_key", os.getenv("AZURE_OPENAI_API_KEY"))
            self.api_version = api_version or config.get("api_version", "2024-12-01-preview")
            self.deployment = deployment or config.get("deployment", "gpt-4.1")
            self.model = model or config.get("model", "gpt-4.1")
        else:
            # Fall back to environment variables or provided values
            self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "https://clinicalml-cloudbank-openai.openai.azure.com/")
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
            self.model = model or os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1")
        
        # Validate that we have required credentials
        if not self.api_key or self.api_key == "your-api-key-here":
            raise ValueError("Azure OpenAI API key not found! Please set AZURE_OPENAI_API_KEY environment variable or create azure_config.json")
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p


class SFTExample:
    """Data structure for supervised fine-tuning examples"""
    
    def __init__(self,
                 patient_id: int,
                 label_time: str,
                 label_value: bool,
                 input_text: str,
                 target_text: str,
                 task_instruction: str,
                 backwards_reasoning: Optional[str] = None,
                 final_prediction: Optional[str] = None,
                 backwards_prompt: Optional[str] = None,
                 prompt_tokens: Optional[int] = None,
                 completion_tokens: Optional[int] = None,
                 total_tokens: Optional[int] = None):
        self.patient_id = patient_id
        self.label_time = label_time
        self.label_value = label_value
        self.input_text = input_text
        self.target_text = target_text
        self.task_instruction = task_instruction
        self.backwards_reasoning = backwards_reasoning
        self.final_prediction = final_prediction
        self.backwards_prompt = backwards_prompt
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
    
    def to_dict(self):
        return {
            'patient_id': self.patient_id,
            'label_time': self.label_time,
            'label_value': self.label_value,
            'input_text': self.input_text,
            'target_text': self.target_text,
            'task_instruction': self.task_instruction,
            'backwards_reasoning': self.backwards_reasoning,
            'final_prediction': self.final_prediction,
            'backwards_prompt': self.backwards_prompt,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens
        }


class AzureBackwardsReasoningGenerator:
    """Generates backwards reasoning traces using Azure OpenAI"""
    
    def __init__(self, config: AzureBackwardsReasoningConfig):
        self.config = config
        self.client = AzureOpenAI(
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
        )
    
    def _create_backwards_prompt(self, input_text: str, ground_truth: str, task_instruction: str) -> str:
        """Create backwards reasoning prompt with ground truth"""
        
        # Determine the correct answer format
        correct_answer = "Positive" if ground_truth == "Positive" else "Negative"
        
        return f"""You are a medical expert tasked with explaining clinical reasoning. You have been given a patient's electronic healthcare record (EHR) in Markdown format and the correct prediction outcome. Your task is to provide a detailed, step-by-step reasoning process that explains why this outcome is correct, WITHOUT explicitly stating that you know the answer in advance.

Focus on identifying the key clinical indicators, risk factors, and evidence from the patient's medical history that support the correct prediction. Structure your reasoning logically and provide specific examples from the patient data.

Only after providing your reasoning, conclude with your final answer (Positive or Negative) in the following format:
Final Answer: [Positive/Negative]

Patient's Electronic Healthcare Record:
{input_text}

Task: {task_instruction}

Correct Prediction: {correct_answer}

Now, provide a detailed reasoning process that explains why this prediction is correct based on the clinical evidence:

Reasoning:"""
    
    
    
    def _generate_response(self, prompt: str) -> tuple:
        """Generate response from Azure OpenAI and return content with usage info"""
        try:
            # Build base parameters
            completion_params = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical expert AI assistant that provides detailed clinical reasoning for medical predictions. You excel at analyzing patient data and explaining clinical decision-making processes.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "model": self.config.deployment
            }
            
            # Only include top_p if the model supports it (some models like gpt-5-mini don't)
            if "gpt-5" not in self.config.deployment.lower():
                completion_params["top_p"] = self.config.top_p
            
            response = self.client.chat.completions.create(**completion_params)
            
            content = response.choices[0].message.content.strip()
            usage = {
                'prompt_tokens': response.usage.prompt_tokens if response.usage else None,
                'completion_tokens': response.usage.completion_tokens if response.usage else None,
                'total_tokens': response.usage.total_tokens if response.usage else None
            }
            
            return content, usage
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}", {'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None}
    
    def generate_backwards_reasoning(self, example: SFTExample) -> SFTExample:
        """Generate backwards reasoning using ground truth"""
        
        # Generate backwards reasoning only
        backwards_prompt = self._create_backwards_prompt(
            example.input_text, 
            example.target_text, 
            example.task_instruction
        )
        
        reasoning_response, reasoning_usage = self._generate_response(backwards_prompt)
        
        # Update example with results
        example.backwards_reasoning = reasoning_response
        example.final_prediction = example.target_text  # Use ground truth as final prediction
        example.backwards_prompt = backwards_prompt
        
        # Use only reasoning token usage
        example.prompt_tokens = reasoning_usage['prompt_tokens'] or 0
        example.completion_tokens = reasoning_usage['completion_tokens'] or 0
        example.total_tokens = reasoning_usage['total_tokens'] or 0
        
        return example


class AzureBackwardsReasoningPipeline:
    """Main pipeline for generating backwards reasoning traces for SFT"""
    
    def __init__(self, config: AzureBackwardsReasoningConfig):
        self.config = config
        self.generator = AzureBackwardsReasoningGenerator(config)
    
    def load_examples_from_jsonl(self, data_dir: str):
        """Load examples from JSONL files in data directory"""
        examples = []
        
        # Load train data
        train_file = os.path.join(data_dir, "train.jsonl")
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    example = SFTExample(
                        patient_id=data['patient_id'],
                        label_time=data['label_time'],
                        label_value=data['label_value'],
                        input_text=data['input_text'],
                        target_text=data['target_text'],
                        task_instruction=data.get('task_instruction', '')
                    )
                    examples.append(example)
        else:
            logger.error(f"Training data file not found: {train_file}")
        
        logger.info(f"Loaded {len(examples)} examples from {data_dir}")
        return examples
    
    def load_examples_from_database(self, path_to_database: str, path_to_labels_dir: str, 
                                  task_to_instructions: str = "", num_samples: int = None):
        """Load examples directly from database"""
        # Import here to avoid issues when femr is not available
        try:
            # Add parent directory to path for imports
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from llm_featurizer import load_labeled_patients_with_tasks, LLMFeaturizer, preprocess_llm_featurizer
            from serialization.ehr_serializer import UniqueThenListVisitsWOAllCondsWithValuesStrategy
            from femr.extension import datasets as extension_datasets
            import collections
        except ImportError as e:
            logger.error(f"Required modules not available: {e}")
            logger.error("Please use --data_dir option with pre-generated JSONL files instead")
            return []
        
        PatientDatabase = extension_datasets.PatientDatabase
        
        # Load instructions
        task_to_instructions_dict = {}
        if task_to_instructions and os.path.exists(task_to_instructions):
            with open(task_to_instructions, 'r') as f:
                task_to_instructions_dict = json.load(f)
        
        # Load labels for the specific task (acute MI)
        # Check if path_to_labels_dir is a single task directory or contains multiple tasks
        if os.path.exists(os.path.join(path_to_labels_dir, 'labeled_patients.csv')):
            # Single task directory
            task_dirs = [os.path.basename(path_to_labels_dir)]
            labels_base_dir = os.path.dirname(path_to_labels_dir)
        else:
            # Multiple task directories
            task_dirs = [d for d in os.listdir(path_to_labels_dir) 
                         if os.path.isdir(os.path.join(path_to_labels_dir, d))]
            labels_base_dir = path_to_labels_dir
        
        logger.info(f"Found {len(task_dirs)} task directories: {task_dirs}")
        
        # Import LabelTask
        from llm_featurizer import LabelTask
        
        # Load and combine labels from all tasks
        all_patients_to_labels: Dict[int, List[LabelTask]] = collections.defaultdict(list)
        for task_dir in task_dirs:
            label_file = os.path.join(labels_base_dir, task_dir, 'labeled_patients.csv')
            if not os.path.exists(label_file):
                logger.warning(f"Label file not found for task {task_dir}: {label_file}")
                continue
                
            task_patients_to_labels = load_labeled_patients_with_tasks(label_file)
            for patient_id, labels in task_patients_to_labels.items():
                all_patients_to_labels[patient_id].extend(labels)
        
        logger.info(f"Loaded labels from {len(task_dirs)} tasks")
        logger.info(f"Total patients: {len(all_patients_to_labels)}")
        logger.info(f"Total labels: {sum(len(labels) for labels in all_patients_to_labels.values())}")
        
        # Load database
        logger.info(f"Loading database from {path_to_database}")
        database = PatientDatabase(path_to_database)
        
        # Create serialization strategy
        serialization_strategy = UniqueThenListVisitsWOAllCondsWithValuesStrategy(
            num_aggregated_events=3
        )
        
        # Initialize LLMFeaturizer
        llm_featurizer = LLMFeaturizer(
            embedding_size=1,  # Not used for text generation
            serialization_strategy=serialization_strategy,
            task_to_instructions=task_to_instructions_dict,
            excluded_ontologies=['LOINC', 'Domain', 'CARE_SITE', 'ICDO3', 'Medicare Specialty', 'CMS Place of Service', 'OMOP Extension', 'Condition Type'],
            filter_aggregated_events=True,
            time_window=None
        )
        
        # Preprocess featurizer
        logger.info("Start | Preprocess featurizers")
        llm_featurizer = preprocess_llm_featurizer(
            path_to_database, 
            llm_featurizer, 
            all_patients_to_labels, 
            1  # num_threads
        )
        logger.info("Finish | Preprocess featurizers")
        
        # Convert to SFTExample objects
        examples = []
        for (pid, label_idx), (instruction, text) in llm_featurizer.pid_label_idx_serializations.items():
            label = all_patients_to_labels[pid][label_idx]
            
            # Create input text with task instruction
            input_text = f"{instruction}\n\nPatient Medical History:\n{text}"
            
            # Create target text
            target_text = "Positive" if label.value else "Negative"
            
            example = SFTExample(
                patient_id=pid,
                label_time=label.time.isoformat(),
                label_value=label.value,
                input_text=input_text,
                target_text=target_text,
                task_instruction=instruction
            )
            examples.append(example)
        
        # Limit samples if specified
        if num_samples and num_samples < len(examples):
            examples = examples[:num_samples]
            logger.info(f"Limited to {num_samples} samples")
        
        logger.info(f"Created {len(examples)} SFT examples from database")
        return examples
    
    def print_backwards_reasoning(self, example: SFTExample):
        """Print backwards reasoning for a specific example"""
        print("=" * 80)
        print(f"PATIENT ID: {example.patient_id}")
        print(f"TASK: {example.task_instruction}")
        print(f"GROUND TRUTH: {example.target_text}")
        print("=" * 80)
        
        # Generate backwards reasoning
        example = self.generator.generate_backwards_reasoning(example)
        
        print(f"\nFINAL PREDICTION: {example.final_prediction}")
        print(f"CORRECT: {example.final_prediction == example.target_text}")
        print("\n" + "-" * 40 + " BACKWARDS REASONING " + "-" * 40)
        print(example.backwards_reasoning)
        
        print("\n" + "=" * 80 + "\n")
    
    def print_multiple_traces(self, examples: List[SFTExample], 
                            patient_ids: Optional[List[int]] = None):
        """Print backwards reasoning for multiple examples"""
        if patient_ids:
            # Filter examples by patient IDs
            filtered_examples = [ex for ex in examples if ex.patient_id in patient_ids]
            if not filtered_examples:
                logger.warning(f"No examples found for patient IDs: {patient_ids}")
                return
            examples = filtered_examples
        
        for i, example in enumerate(examples):
            print(f"\nProcessing example {i+1}/{len(examples)}...")
            self.print_backwards_reasoning(example)
    
    def save_sft_traces_incremental(self, example: SFTExample, output_file: str, is_first: bool = False):
        """Save a single SFT reasoning trace incrementally to JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # If this is the first example, create a new file with array structure
        if is_first:
            with open(output_file, 'w') as f:
                json.dump([example.to_dict()], f, indent=2)
        else:
            # Read existing data, append new example, and write back
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # If file doesn't exist or is corrupted, start fresh
                data = []
            
            data.append(example.to_dict())
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved patient {example.patient_id} SFT reasoning trace to {output_file}")
    
    def save_sft_dataset_incremental(self, example: SFTExample, output_file: str, is_first: bool = False):
        """Save a single SFT dataset entry incrementally to JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create the conversation format for this single example
        conversation = [
            {
                "role": "user",
                "content": f"Given a patient's electronic healthcare record (EHR) in Markdown format, {example.task_instruction}\n\nPatient Medical History:\n{example.input_text.split('Patient Medical History:')[1] if 'Patient Medical History:' in example.input_text else example.input_text}\n\nPlease provide your analysis, and conclude with your final prediction in the format Final Answer: [Positive/Negative]."
            },
            {
                "role": "assistant", 
                "content": f"{example.backwards_reasoning}"
            }
        ]
        
        sft_entry = {
            "conversations": conversation,
            "patient_id": example.patient_id,
            "label_time": example.label_time,
            "label_value": example.label_value,
            "task_instruction": example.task_instruction
        }
        
        # If this is the first example, create a new file with array structure
        if is_first:
            with open(output_file, 'w') as f:
                json.dump([sft_entry], f, indent=2)
        else:
            # Read existing data, append new entry, and write back
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # If file doesn't exist or is corrupted, start fresh
                data = []
            
            data.append(sft_entry)
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved patient {example.patient_id} SFT dataset entry to {output_file}")
    


def parse_args():
    parser = argparse.ArgumentParser(description="Azure-based backwards reasoning pipeline for SFT")
    
    # Data source options (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data_dir", type=str, 
                           help="Directory containing pre-generated JSONL files (train.jsonl)")
    data_group.add_argument("--database", type=str, 
                           help="Load directly from database (like existing pipeline)")
    
    # Database-specific arguments (when using --database)
    parser.add_argument("--path_to_labels_dir", type=str, 
                       help="Path to labels directory (required with --database)")
    parser.add_argument("--task_to_instructions", type=str, default="", 
                       help="Path to task instructions file")
    
    # General arguments
    parser.add_argument("--patient_ids", type=str, nargs="+", 
                       help="Specific patient IDs to process (space-separated)")
    parser.add_argument("--max_examples", type=int, default=10, 
                       help="Maximum number of examples to process")
    parser.add_argument("--output_file", type=str, 
                       help="Output file to save SFT reasoning traces (optional)")
    parser.add_argument("--sft_dataset_file", type=str,
                       help="Output file to save SFT dataset in conversation format")
    parser.add_argument("--temperature", type=float, default=0.3, 
                       help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, 
                       help="Maximum tokens to generate")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate arguments
    if args.database and not args.path_to_labels_dir:
        logger.error("--path_to_labels_dir is required when using --database")
        return
    
    # Create configuration
    config = AzureBackwardsReasoningConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Initialize pipeline
    pipeline = AzureBackwardsReasoningPipeline(config)
    
    # Load examples based on mode
    if args.data_dir:
        # Load from pre-generated JSONL files
        logger.info("Loading examples from JSONL files...")
        examples = pipeline.load_examples_from_jsonl(args.data_dir)
    else:
        # Load directly from database
        logger.info("Loading examples directly from database...")
        examples = pipeline.load_examples_from_database(
            args.database,
            args.path_to_labels_dir,
            args.task_to_instructions,
            args.max_examples
        )
    
    if not examples:
        logger.error("No examples loaded. Exiting.")
        return
    
    # Limit examples if specified
    if args.max_examples and args.max_examples < len(examples):
        examples = examples[:args.max_examples]
        logger.info(f"Limited to {args.max_examples} examples")
    
    # Convert patient IDs to integers if provided
    patient_ids = None
    if args.patient_ids:
        try:
            patient_ids = [int(pid) for pid in args.patient_ids]
        except ValueError:
            logger.error("Patient IDs must be integers")
            return
    
    # Generate backwards reasoning
    logger.info("Generating backwards reasoning traces...")
    
    # Track total tokens used
    total_tokens = 0
    
    # Generate reasoning for all examples
    for i, example in enumerate(examples):
        logger.info(f"Processing example {i+1}/{len(examples)} (Patient {example.patient_id})...")
        example = pipeline.generator.generate_backwards_reasoning(example)
        
        # Save after each patient if output file is specified
        if args.output_file:
            is_first = (i == 0)
            pipeline.save_sft_traces_incremental(example, args.output_file, is_first)
        
        # Save SFT dataset incrementally if specified
        if args.sft_dataset_file:
            is_first = (i == 0)
            pipeline.save_sft_dataset_incremental(example, args.sft_dataset_file, is_first)
        
        # Add tokens from this example to total
        if example.total_tokens:
            total_tokens += example.total_tokens
            logger.info(f"Example {i+1} used {example.total_tokens} tokens (Total so far: {total_tokens})")
        else:
            logger.warning(f"Example {i+1} - token count not available")
    
    logger.info(f"Total tokens used across all examples: {total_tokens}")
    
    # Print reasoning traces
    pipeline.print_multiple_traces(examples, patient_ids)
    
    # Note: Both SFT traces and SFT dataset are already saved incrementally above, no need to save again
    
    logger.info("Backwards reasoning pipeline completed!")


if __name__ == "__main__":
    main()
