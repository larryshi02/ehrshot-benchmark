#!/usr/bin/env python3
"""
Serialize patient EHRs into text for all 15 clinical prediction tasks.

Inputs:
  --path_to_database   : Path to the FEMR patient database.
  --path_to_labels_dir : Directory containing all_labels_tasks_out.csv.
  --path_to_splits     : CSV mapping patient IDs to train/val/test splits.
  --output_dir         : Where to write per-task per-split JSON files.
  --mode               : "plaintext" (no vitals sections) or
                         "manualrubric" (includes 3 recent-vitals sections).

Outputs:
  {output_dir}/{task}/{split}.json   -- one file per task per split.

Each JSON file is a list of dicts:
  { patient_id, prediction_time, task, split, label, serialization,
    original_tokens, was_clipped }

Cohort balancing (applied per task per split):
  val  : min(50, smallest-class-count) per class  (balanced pos/neg)
  train/test : if total > 3000 =>
      positives < 1000 : match negatives to positives  (balanced)
      positives >= 1000: 1000 each = 2000 total        (balanced)
  Otherwise: keep all.

Token clipping:
  Serializations are clipped to 8192 tokens using the
  Qwen/Qwen3-Embedding-8B tokenizer.

Connects to: 02_create_sft (consumes the JSON files produced here).
"""

import argparse
import os
import sys
import json
import csv
import random
import collections
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from loguru import logger
from transformers import AutoTokenizer
from femr.extension import datasets as extension_datasets
from femr import Patient, Event
from femr.featurizers.featurizers import get_patient_birthdate

# Resolve imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES, TOKENIZER_NAME, MAX_SERIALIZATION_TOKENS, SEED
from ehr_serializer import (
    EHRSerializer,
    AGGREGATED_EVENTS_CODES_LOINC,
)

PatientDatabase = extension_datasets.PatientDatabase
Ontology = extension_datasets.Ontology

RANDOM_SEED = SEED
AGE_IDENTIFIER = "Patient age"

# Excluded ontologies (same as the "no_labs" preset in the original code)
EXCLUDED_ONTOLOGIES = ["LOINC", "Domain", "CARE_SITE", "ICDO3"]

# ---------------------------------------------------------------------------
# Cohort balancing
# ---------------------------------------------------------------------------

def balance_split(records: List[dict], split: str) -> List[dict]:
    """Return a (possibly subsampled) balanced list of records for *split*."""
    rng = random.Random(RANDOM_SEED)
    pos = [r for r in records if r["label"] is True]
    neg = [r for r in records if r["label"] is False]

    if split == "val":
        n = min(50, len(pos), len(neg))
        pos = rng.sample(pos, n)
        neg = rng.sample(neg, n)
    elif split in ("train", "test"):
        total = len(pos) + len(neg)
        if total > 3000:
            if len(pos) < 1000:
                # match negatives to positives
                n = len(pos)
            else:
                n = 1000
            pos = rng.sample(pos, min(n, len(pos)))
            neg = rng.sample(neg, min(n, len(neg)))
    # else: keep all

    result = pos + neg
    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Token clipping
# ---------------------------------------------------------------------------

def clip_text(text: str, tokenizer, max_tokens: int) -> Tuple[str, int, bool]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n = len(tokens)
    if n <= max_tokens:
        return text, n, False
    clipped = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    return clipped, n, True


# ---------------------------------------------------------------------------
# Patient loading
# ---------------------------------------------------------------------------

def load_splits(path: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            mapping[int(row["omop_person_id"])] = row["split"]
    return mapping


def load_labels(path: str):
    patients: Dict[int, List[Tuple[datetime, str]]] = collections.defaultdict(list)
    values: Dict[Tuple[int, datetime, str], bool] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            t = datetime.fromisoformat(row["prediction_time"])
            if t.second != 0:
                t = t.replace(second=0)
            task = row["task"]
            pid = int(row["patient_id"])
            val = row["value"] == "True"
            patients[pid].append((t, task))
            values[(pid, t, task)] = val
    return patients, values


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_patient(
    database: PatientDatabase,
    ontology: Ontology,
    patient_id: int,
    label_time: datetime,
    num_aggregated: int,
) -> str:
    patient: Patient = database[patient_id]

    def is_visit(event: Event) -> bool:
        return event.code.startswith("Visit/")

    def resolve_code(code: str, included_ontologies: List[str] = []) -> Optional[str]:
        ont = code.split("/")[0].strip()
        if (ont in EXCLUDED_ONTOLOGIES
                and code not in AGGREGATED_EVENTS_CODES_LOINC
                and ont not in included_ontologies):
            return None
        if code.startswith(f"{AGE_IDENTIFIER}: "):
            return code
        if ont == "Cancer Modifier" and "OMOP" not in code:
            return code.split("/", 1)[1].replace("_", " ").replace("-", " ").replace("/", " ").strip()
        desc = ontology.get_text_description(code)
        if desc.startswith("Birth"):
            return None
        return desc.strip()

    events = [e for e in patient.events if e.start <= label_time]

    # Replace birth event with age
    birth = get_patient_birthdate(patient)
    age = int((label_time - birth).days / 365)
    if events:
        b = events[0]
        events[0] = Event(b.start, f"{AGE_IDENTIFIER}: {age}", b.value)

    serializer = EHRSerializer()
    serializer.load_from_femr_events(events, resolve_code, is_visit,
                                     filter_aggregated_events=(num_aggregated > 0))
    return serializer.serialize(num_aggregated, label_time=label_time)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path_to_database", required=True)
    p.add_argument("--path_to_labels_dir", required=True)
    p.add_argument("--path_to_splits", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", required=True, choices=["plaintext", "manualrubric"])
    p.add_argument("--tasks", default="", help="Comma-separated task subset (default: all)")
    p.add_argument("--max_tokens", type=int, default=MAX_SERIALIZATION_TOKENS)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    num_aggregated = 0 if args.mode == "plaintext" else 3

    if os.path.exists(args.output_dir) and not args.force:
        raise FileExistsError(f"{args.output_dir} exists. Use --force to overwrite.")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which tasks to process
    task_set = set(args.tasks.split(",")) if args.tasks else set(ALL_TASK_NAMES)
    logger.info(f"Mode={args.mode}  num_aggregated={num_aggregated}  tasks={sorted(task_set)}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

    # Load splits & labels
    patient_to_split = load_splits(args.path_to_splits)
    patients_to_labels, label_values = load_labels(
        os.path.join(args.path_to_labels_dir, "all_labels_tasks_out.csv")
    )
    logger.info(f"Loaded {len(patient_to_split)} splits, "
                f"{len(patients_to_labels)} patients with labels")

    # Filter to requested tasks
    filtered: Dict[int, List[Tuple[datetime, str]]] = {}
    for pid, labels in patients_to_labels.items():
        kept = [(t, task) for t, task in labels if task in task_set]
        if kept:
            filtered[pid] = kept
    patients_to_labels = filtered

    # Load database
    logger.info(f"Loading FEMR database: {args.path_to_database}")
    database = PatientDatabase(args.path_to_database)
    ontology = database.get_ontology()

    # Collect records: {task: {split: [records]}}
    collection: Dict[str, Dict[str, List[dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    total = sum(len(v) for v in patients_to_labels.values())
    done = 0
    clipped_n = 0

    for pid, labels in patients_to_labels.items():
        split = patient_to_split.get(pid)
        if split is None:
            continue
        for label_time, task in labels:
            text = serialize_patient(database, ontology, pid, label_time, num_aggregated)
            clipped_text, orig_tokens, was_clipped = clip_text(text, tokenizer, args.max_tokens)
            if was_clipped:
                clipped_n += 1

            label_val = label_values.get((pid, label_time, task), False)
            collection[task][split].append({
                "patient_id": pid,
                "prediction_time": label_time.isoformat(),
                "task": task,
                "split": split,
                "label": label_val,
                "serialization": clipped_text,
                "original_tokens": orig_tokens,
                "was_clipped": was_clipped,
            })
            done += 1
            if done % 500 == 0:
                logger.info(f"  {done}/{total} serialized (clipped so far: {clipped_n})")

    # Balance & save
    for task in sorted(collection):
        task_dir = os.path.join(args.output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        for split in ("train", "val", "test"):
            records = collection[task].get(split, [])
            if not records:
                continue
            before = len(records)
            records = balance_split(records, split)
            out_path = os.path.join(task_dir, f"{split}.json")
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)
            pos = sum(1 for r in records if r["label"])
            neg = len(records) - pos
            logger.info(f"  {task}/{split}: {before} -> {len(records)} "
                        f"(pos={pos}, neg={neg})  => {out_path}")

    logger.success(f"Done. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
