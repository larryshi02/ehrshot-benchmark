#!/usr/bin/env python3
"""
Clean the training data by removing Unicode artifacts and malformed text.
"""

import json
import re

def clean_training_data(input_file, output_file):
    """Clean the training dataset by removing Unicode artifacts"""
    
    # Load the dataset
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f'Original dataset size: {len(data)}')
    
    # Clean the data
    cleaned_data = []
    artifacts_found = 0
    
    for example in data:
        cleaned_example = example.copy()
        
        # Clean conversations
        for conv in cleaned_example['conversations']:
            original_content = conv['content']
            
            # Clean Unicode artifacts
            cleaned_content = original_content
            cleaned_content = cleaned_content.replace('\uff0c', ',')  # Unicode comma
            cleaned_content = cleaned_content.replace('\u2019', "'")  # Unicode apostrophe
            cleaned_content = cleaned_content.replace('\u2013', '-')  # Unicode dash
            cleaned_content = cleaned_content.replace('\u2014', '--')  # Unicode em dash
            cleaned_content = cleaned_content.replace('\u201c', '"')  # Unicode left quote
            cleaned_content = cleaned_content.replace('\u201d', '"')  # Unicode right quote
            cleaned_content = cleaned_content.replace('\u00b0', ' degrees')  # Degree symbol
            
            # Fix malformed apostrophes and newlines
            cleaned_content = re.sub(r'`([^`]+)`', r"'\1'", cleaned_content)
            cleaned_content = re.sub(r'`\s*', "'", cleaned_content)
            cleaned_content = re.sub(r'\\n', '\n', cleaned_content)
            
            # Remove Chinese characters
            cleaned_content = re.sub(r'[\u4e00-\u9fff]+', '', cleaned_content)
            
            if cleaned_content != original_content:
                artifacts_found += 1
                conv['content'] = cleaned_content
        
        cleaned_data.append(cleaned_example)
    
    print(f'Examples with artifacts found: {artifacts_found}')
    print(f'Cleaned dataset size: {len(cleaned_data)}')
    
    # Save cleaned dataset
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f'Cleaned dataset saved to: {output_file}')
    return cleaned_data

if __name__ == "__main__":
    input_file = "data_gpt-4o/acute_mi_sft_dataset.json"
    output_file = "data_gpt-4o/acute_mi_sft_dataset_cleaned.json"
    
    clean_training_data(input_file, output_file)
