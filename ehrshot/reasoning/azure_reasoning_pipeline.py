#!/usr/bin/env python3
"""
Azure-based reasoning trace pipeline for printing reasoning traces for specific examples.
Uses the Azure OpenAI setup from azure_test.py to generate reasoning traces.
"""

import os
import json
import argparse
from typing import List, Dict, Optional, Any
from openai import AzureOpenAI
from loguru import logger


class AzureReasoningConfig:
    """Configuration for Azure-based reasoning trace generation"""
    
    def __init__(self,
                 endpoint: str = None,
                 api_key: str = None,
                 api_version: str = "2024-12-01-preview",
                 deployment: str = "gpt-4o",
                 model: str = "gpt-4o",
                 max_tokens: int = 4096,
                 temperature: float = 0.7,
                 top_p: float = 1.0):
        
        # Try to load from config file first, then fall back to environment variables
        import os
        import json
        
        config_file = os.path.join(os.path.dirname(__file__), "azure_config.json")
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.endpoint = endpoint or config.get("endpoint", "https://clinicalml-cloudbank-openai.openai.azure.com/")
            self.api_key = api_key or config.get("api_key", os.getenv("AZURE_OPENAI_API_KEY"))
            self.api_version = api_version or config.get("api_version", "2024-12-01-preview")
            self.deployment = deployment or config.get("deployment", "gpt-4o")
            self.model = model or config.get("model", "gpt-4o")
        else:
            # Fall back to environment variables or provided values
            self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "https://clinicalml-cloudbank-openai.openai.azure.com/")
            self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
            self.model = model or os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
        
        # Validate that we have required credentials
        if not self.api_key or self.api_key == "your-api-key-here":
            raise ValueError("Azure OpenAI API key not found! Please set AZURE_OPENAI_API_KEY environment variable or create azure_config.json")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p


class ReasoningExample:
    """Data structure for reasoning trace examples"""
    
    def __init__(self,
                 patient_id: int,
                 label_time: str,
                 label_value: bool,
                 input_text: str,
                 target_text: str,
                 full_sequence: str,
                 task_instruction: str,
                 reasoning_trace: Optional[str] = None,
                 prediction: Optional[str] = None,
                 is_correct: Optional[bool] = None,
                 prompt_tokens: Optional[int] = None,
                 completion_tokens: Optional[int] = None,
                 total_tokens: Optional[int] = None):
        self.patient_id = patient_id
        self.label_time = label_time
        self.label_value = label_value
        self.input_text = input_text
        self.target_text = target_text
        self.full_sequence = full_sequence
        self.task_instruction = task_instruction
        self.reasoning_trace = reasoning_trace
        self.prediction = prediction
        self.is_correct = is_correct
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
            'full_sequence': self.full_sequence,
            'task_instruction': self.task_instruction,
            'reasoning_trace': self.reasoning_trace,
            'prediction': self.prediction,
            'is_correct': self.is_correct,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens
        }


class AzureReasoningGenerator:
    """Generates reasoning traces using Azure OpenAI"""
    
    def __init__(self, config: AzureReasoningConfig):
        self.config = config
        self.client = AzureOpenAI(
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
        )
    
    def _create_base_prompt(self, input_text: str, task_instruction: str) -> str:
        """Create the base prompt without ground truth"""
        return f"""Given a patient's electronic healthcare record (EHR) in Markdown format, {task_instruction}

Patient Medical History:
{input_text}

Please provide a detailed reasoning trace explaining your prediction, then conclude with your final answer (Positive or Negative).

Reasoning:"""
    
    
    def _extract_prediction(self, response: str) -> str:
        """Extract prediction from model response"""
        response_lower = response.lower().strip()
        
        # Look for "positive" or "negative" in the response
        if "positive" in response_lower:
            return "Positive"
        elif "negative" in response_lower:
            return "Negative"
        
        # Look for "yes" or "no" as fallback
        if "yes" in response_lower:
            return "Positive"
        elif "no" in response_lower:
            return "Negative"
        
        # Default to negative if unclear
        return "Negative"
    
    def _generate_response(self, prompt: str) -> tuple[str, dict]:
        """Generate response from Azure OpenAI and return content with usage info"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful medical AI assistant that provides detailed reasoning for clinical predictions.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                model=self.config.deployment
            )
            
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
    
    def generate_reasoning(self, example: ReasoningExample) -> ReasoningExample:
        """Generate reasoning using base prompt strategy"""
        # Create base prompt
        base_prompt = self._create_base_prompt(example.input_text, example.task_instruction)
        
        # Generate response with usage info
        response, usage = self._generate_response(base_prompt)
        prediction = self._extract_prediction(response)
        
        # Check if correct
        is_correct = (prediction == example.target_text)
        
        example.reasoning_trace = response
        example.prediction = prediction
        example.is_correct = is_correct
        example.prompt_tokens = usage['prompt_tokens']
        example.completion_tokens = usage['completion_tokens']
        example.total_tokens = usage['total_tokens']
        
        return example


class AzureReasoningPipeline:
    """Main pipeline for generating and printing reasoning traces"""
    
    def __init__(self, config: AzureReasoningConfig):
        self.config = config
        self.generator = AzureReasoningGenerator(config)
    
    def load_examples_from_jsonl(self, data_dir: str) -> List[ReasoningExample]:
        """Load examples from JSONL files in data directory"""
        examples = []
        
        # Load train data
        train_file = os.path.join(data_dir, "train.jsonl")
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    example = ReasoningExample(
                        patient_id=data['patient_id'],
                        label_time=data['label_time'],
                        label_value=data['label_value'],
                        input_text=data['input_text'],
                        target_text=data['target_text'],
                        full_sequence=data['full_sequence'],
                        task_instruction=data.get('task_instruction', '')
                    )
                    examples.append(example)
        else:
            logger.error(f"Training data file not found: {train_file}")
        
        logger.info(f"Loaded {len(examples)} examples from {data_dir}")
        return examples
    
    def load_examples_from_database(self, path_to_database: str, path_to_labels_dir: str, 
                                  task_to_instructions: str = "", num_samples: int = None) -> List[ReasoningExample]:
        """Load examples directly from database (like existing pipeline)"""
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
        
        # Load labels for all tasks
        task_dirs = [d for d in os.listdir(path_to_labels_dir) 
                     if os.path.isdir(os.path.join(path_to_labels_dir, d))]
        
        logger.info(f"Found {len(task_dirs)} task directories: {task_dirs}")
        
        # Load and combine labels from all tasks
        all_patients_to_labels: Dict[int, List[Tuple[datetime, str]]] = collections.defaultdict(list)
        for task_dir in task_dirs:
            label_file = os.path.join(path_to_labels_dir, task_dir, 'labeled_patients.csv')
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
        
        # Convert to ReasoningExample objects
        examples = []
        for (pid, label_idx), (instruction, text) in llm_featurizer.pid_label_idx_serializations.items():
            label_time, label_value = all_patients_to_labels[pid][label_idx]
            
            # Create input text with task instruction
            input_text = f"Given a patient's electronic healthcare record (EHR) in Markdown format, {instruction}\n\nPatient Medical History:\n{text}\n\nPrediction: "
            
            # Create target text
            target_text = "Positive" if label_value else "Negative"
            
            # Create full sequence
            full_sequence = input_text + target_text
            
            example = ReasoningExample(
                patient_id=pid,
                label_time=label_time.isoformat(),
                label_value=label_value,
                input_text=input_text,
                target_text=target_text,
                full_sequence=full_sequence,
                task_instruction=instruction
            )
            examples.append(example)
        
        # Limit samples if specified
        if num_samples and num_samples < len(examples):
            examples = examples[:num_samples]
            logger.info(f"Limited to {num_samples} samples")
        
        logger.info(f"Created {len(examples)} reasoning examples from database")
        return examples
    
    def print_reasoning_trace(self, example: ReasoningExample):
        """Print reasoning trace for a specific example"""
        print("=" * 80)
        print(f"PATIENT ID: {example.patient_id}")
        print(f"TASK: {example.task_instruction}")
        print(f"GROUND TRUTH: {example.target_text}")
        print("=" * 80)
        
        # Generate reasoning
        example = self.generator.generate_reasoning(example)
        
        print(f"\nPREDICTION: {example.prediction}")
        print(f"CORRECT: {example.is_correct}")
        print("\n" + "-" * 40 + " REASONING TRACE " + "-" * 40)
        print(example.reasoning_trace)
        
        print("\n" + "=" * 80 + "\n")
    
    def print_multiple_traces(self, examples: List[ReasoningExample], 
                            patient_ids: Optional[List[int]] = None):
        """Print reasoning traces for multiple examples"""
        if patient_ids:
            # Filter examples by patient IDs
            filtered_examples = [ex for ex in examples if ex.patient_id in patient_ids]
            if not filtered_examples:
                logger.warning(f"No examples found for patient IDs: {patient_ids}")
                return
            examples = filtered_examples
        
        for i, example in enumerate(examples):
            print(f"\nProcessing example {i+1}/{len(examples)}...")
            self.print_reasoning_trace(example)
    
    def save_reasoning_traces(self, examples: List[ReasoningExample], 
                            output_file: str):
        """Save reasoning traces to JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert all examples to a list of dictionaries
        data = [example.to_dict() for example in examples]
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(examples)} reasoning traces to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Azure-based reasoning trace pipeline")
    
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
    parser.add_argument("--max_examples", type=int, default=5, 
                       help="Maximum number of examples to process")
    parser.add_argument("--output_file", type=str, 
                       help="Output file to save reasoning traces (optional)")
    parser.add_argument("--temperature", type=float, default=0.7, 
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
    config = AzureReasoningConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Initialize pipeline
    pipeline = AzureReasoningPipeline(config)
    
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
    
    # Generate and print reasoning traces
    logger.info("Generating reasoning traces...")
    
    # Generate reasoning traces
    for example in examples:
        example = pipeline.generator.generate_reasoning(example)
    
    # Print reasoning traces
    pipeline.print_multiple_traces(examples, patient_ids)
    
    # Save to file if requested
    if args.output_file:
        pipeline.save_reasoning_traces(examples, args.output_file)
    
    logger.info("Pipeline completed!")


if __name__ == "__main__":
    main()
