"""
Shared task configuration constants for evaluation scripts.

This module contains task-specific configurations that are shared across
multiple evaluation scripts, including task name mappings and queries.
"""

# Task name mappings (internal task names to dataset names)
TASK_MAPPINGS = {
    'acute_mi': 'new_acutemi',
    'hyperlipidemia': 'new_hyperlipidemia',
    'hypertension': 'new_hypertension',
    'pancreatic_cancer': 'new_pancan'
}

# Task-specific queries for prompt formatting
TASK_QUERIES = {
    'acute_mi': 'will the patient develop an acute myocardial infarction in the next year',
    'hyperlipidemia': 'will the patient develop hyperlipidemia in the next year',
    'hypertension': 'will the patient develop hypertension in the next year',
    'pancreatic_cancer': 'will the patient develop pancreatic cancer in the next year'
}

# Default list of tasks
DEFAULT_TASKS = ['acute_mi', 'hyperlipidemia', 'hypertension', 'pancreatic_cancer']

