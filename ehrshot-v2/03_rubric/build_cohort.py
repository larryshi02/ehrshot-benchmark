#!/usr/bin/env python3
"""
Build diverse 40-patient cohorts per task via label-stratified k-means.

For each task, this script:
  1. Loads training-split serialized records (from 01_serialize plaintext).
  2. Embeds every patient with Qwen/Qwen3-Embedding-8B.
  3. Runs k-means (k=20) separately on positives and negatives.
  4. Selects the medoid of each cluster -> 20 pos + 20 neg = 40 patients.
  5. Saves the cohort JSON and a flat list of patient IDs.

The same 40 patient IDs are reused for:
  - Rubric creation  (03_rubric/create_rubric.py)
  - n=40 fine-tuning experiments (05_train)

Inputs:
  --input_dir  : plaintext serialized directory (data/serialized/plaintext).
  --output_dir : where to write cohort files.
  --tasks      : space-separated list of tasks (default: all 15).

Outputs:
  {output_dir}/{task}/cohort.json     -- full cohort records (40 items)
  {output_dir}/{task}/patient_ids.json -- just the 40 patient_id ints

Connects to:
  - Upstream : 01_serialize (plaintext)
  - Downstream : create_rubric.py, apply_rubric.py, 05_train (n=40)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import TASKS, ALL_TASK_NAMES, EMBEDDING_MODEL, SEED


# ---------------------------------------------------------------------------
# Embedding encoder
# ---------------------------------------------------------------------------

class _TextDataset(TorchDataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]


class EmbeddingEncoder:
    """Qwen3-Embedding-8B encoder with last-token pooling."""

    def __init__(self, model_name: str = EMBEDDING_MODEL,
                 max_length: int = 8192, batch_size: int = 4):
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device).eval()

    @staticmethod
    def _last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask[:, -1].sum() == mask.shape[0]:
            return hidden[:, -1]
        seq_len = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_len]

    @torch.no_grad()
    def encode(self, texts: List[str],
               instruction: str = "") -> np.ndarray:
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery:\n{t}" for t in texts]
        loader = DataLoader(_TextDataset(texts), batch_size=self.batch_size,
                            shuffle=False)
        parts = []
        for batch in tqdm(loader, desc="Embedding"):
            tok = self.tokenizer(batch, max_length=self.max_length,
                                 padding=True, truncation=True,
                                 return_tensors="pt").to(self.device)
            out = self.model(**tok)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            emb = self._last_token_pool(h, tok["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1).cpu().numpy()
            parts.append(emb)
        return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# K-means + medoid selection
# ---------------------------------------------------------------------------

def _select_medoids(X: np.ndarray, km: KMeans,
                    indices: np.ndarray) -> List[int]:
    selected = []
    for cid in range(km.n_clusters):
        mask = km.labels_ == cid
        cidx = np.where(mask)[0]
        if len(cidx) == 0:
            continue
        dists = np.linalg.norm(X[cidx] - km.cluster_centers_[cid], axis=1)
        selected.append(int(indices[cidx[np.argmin(dists)]]))
    return selected


def build_cohort_for_task(
    records: List[dict],
    encoder: EmbeddingEncoder,
    task_query: str,
    n_per_class: int = 20,
) -> Tuple[List[dict], List[int]]:
    pos = [r for r in records if r["label"] is True]
    neg = [r for r in records if r["label"] is False]
    logger.info(f"  train: {len(pos)} pos, {len(neg)} neg")

    all_texts = [r["serialization"] for r in records]
    embeddings = encoder.encode(all_texts, instruction=task_query)

    pos_mask = np.array([r["label"] is True for r in records])
    selected_records: List[dict] = []

    for subset, mask_val, label in [(pos, True, "pos"), (neg, False, "neg")]:
        n = min(n_per_class, len(subset))
        if n == 0:
            continue
        mask = pos_mask == mask_val
        sub_emb = embeddings[mask]
        sub_idx = np.arange(len(records))[mask]
        km = KMeans(n_clusters=n, random_state=SEED, n_init=10)
        km.fit(sub_emb)
        medoid_idx = _select_medoids(sub_emb, km, sub_idx)
        for i in medoid_idx:
            selected_records.append(records[i])
        logger.info(f"  selected {len(medoid_idx)} {label} medoids")

    patient_ids = [r["patient_id"] for r in selected_records]
    return selected_records, patient_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True,
                   help="Plaintext serialized dir (data/serialized/plaintext)")
    p.add_argument("--output_dir", required=True,
                   help="Where to write cohort JSONs")
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    p.add_argument("--n_per_class", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    # Seed for deterministic embedding inference and k-means
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    encoder = EmbeddingEncoder(batch_size=args.batch_size)

    for task in args.tasks:
        logger.info(f"\n{'='*60}\nBuilding cohort for: {task}\n{'='*60}")
        train_path = Path(args.input_dir) / task / "train.json"
        if not train_path.exists():
            logger.warning(f"  no train.json at {train_path}, skipping")
            continue
        with open(train_path) as f:
            records = json.load(f)

        cohort, pids = build_cohort_for_task(
            records, encoder, TASKS[task], args.n_per_class)

        out_dir = Path(args.output_dir) / task
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "cohort.json", "w") as f:
            json.dump(cohort, f, indent=2)
        with open(out_dir / "patient_ids.json", "w") as f:
            json.dump(pids, f)
        logger.info(f"  saved {len(cohort)} records -> {out_dir}")

    logger.success("All cohorts built.")


if __name__ == "__main__":
    main()
