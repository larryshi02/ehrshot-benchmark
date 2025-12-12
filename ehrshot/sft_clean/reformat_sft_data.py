import json
import os
import glob
import re

BASE_DIR = "/home/demirel/clinical-reasoning/ehrshot-benchmark/ehrshot/sft_clean/data_gpt-5-mini/sft_rt_yn"
SUBFOLDERS = ["train_all", "train_orig", "val_small"]

def format_conversation(entry):
    patient_id = entry.get("patient_id")
    label_time = entry.get("label_time")
    label_value = entry.get("label_value")
    input_text = entry.get("input_text", "").strip()
    task_instruction = entry.get("task_instruction", "")
    backwards_reasoning = entry.get("backwards_reasoning", "")
    final_prediction = entry.get("final_prediction", "")
    
    # Clean up input_text
    # It usually starts with headers, we just ensure it's clean.
    
    # Process backwards_reasoning to separate reasoning from Final Answer
    # Expected format ends with "Final Answer: [Positive/Negative]<STOP>" or similar
    # We want to extract the reasoning part.
    
    reasoning_content = backwards_reasoning
    
    # Regex to find Final Answer at the end
    # We look for "Final Answer:" followed by anything until end or <STOP>
    match = re.search(r'(?s)(.*?)(Final Answer:.*$)', backwards_reasoning)
    if match:
        reasoning_content = match.group(1).strip()
        # The remainder is match.group(2), which is the final answer line.
    else:
        # If not found, we assume the whole thing is reasoning? 
        # Or maybe it's just missing. We'll use the whole thing but warn if needed.
        # But we also have `final_prediction` field.
        pass

    # Construct System Message
    system_content = (
        f"You are a medical expert. You will be provided with a patient's Electronic Healthcare Record (EHR).\n"
        f"Your task is to predict: {task_instruction}\n"
        f"Think step-by-step to analyze the patient's condition and risk factors.\n"
        f"Format your response as follows:\n"
        f"<think>\n"
        f"[Detailed clinical reasoning]\n"
        f"</think>\n\n"
        f"Final Answer: [Positive/Negative]"
    )

    # Construct User Message
    user_content = input_text

    # Construct Assistant Message
    # We assume final_prediction is e.g. "Negative" or "Positive"
    assistant_content = (
        f"<think>\n"
        f"{reasoning_content}\n"
        f"</think>\n\n"
        f"Final Answer: {final_prediction}"
    )

    new_conversations = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": assistant_content
        }
    ]
    
    new_entry = {
        "conversations": new_conversations,
        "patient_id": patient_id,
        "label_time": label_time,
        "label_value": label_value,
        "task_instruction": task_instruction
    }
    return new_entry

def process_folder(folder_path):
    print(f"Processing folder: {folder_path}")
    # Find all *_backwards_reasoning.json files
    pattern = os.path.join(folder_path, "*_backwards_reasoning.json")
    files = glob.glob(pattern)
    
    for br_file in files:
        # Determine output filename
        # e.g. acute_mi_backwards_reasoning.json -> acute_mi_sft_dataset.json
        # Handle "balanced" naming in val_small if present (e.g. acute_mi_balanced_backwards_reasoning.json)
        
        filename = os.path.basename(br_file)
        if "_backwards_reasoning.json" in filename:
            output_filename = filename.replace("_backwards_reasoning.json", "_sft_dataset.json")
            output_path = os.path.join(folder_path, output_filename)
            
            print(f"  Reading {filename} -> Writing {output_filename}")
            
            with open(br_file, 'r') as f:
                data = json.load(f)
            
            new_data = []
            for entry in data:
                new_entry = format_conversation(entry)
                new_data.append(new_entry)
            
            with open(output_path, 'w') as f:
                json.dump(new_data, f, indent=4)
        else:
            print(f"Skipping file with unexpected name pattern: {filename}")

def main():
    for sub in SUBFOLDERS:
        folder_path = os.path.join(BASE_DIR, sub)
        if os.path.exists(folder_path):
            process_folder(folder_path)
        else:
            print(f"Subfolder not found: {folder_path}")

if __name__ == "__main__":
    main()
