# src/preprocess.py
"""GSM8K preprocessing pipeline – zero-leakage."""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List

import datasets
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

CACHE_DIR = Path(".cache").resolve()

datasets.config.HF_DATASETS_CACHE = str(CACHE_DIR)


def _proc(example, tokenizer, cfg):
    q = example[cfg.dataset.text_column].strip()
    a = example[cfg.dataset.label_column].strip()
    prompt = f"### Question:\n{q}\n### Answer:\n"
    p_ids = tokenizer(prompt, truncation=True, max_length=cfg.dataset.max_length, add_special_tokens=False)[
        "input_ids"
    ]
    a_ids = tokenizer(a + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    placeholder = [tokenizer.pad_token_id] * len(a_ids)
    input_ids = (p_ids + placeholder)[: cfg.dataset.max_length]
    labels = ([-100] * len(p_ids) + a_ids)[: cfg.dataset.max_length]
    attn = [1] * len(input_ids)
    pad_len = cfg.dataset.max_length - len(input_ids)
    input_ids.extend([tokenizer.pad_token_id] * pad_len)
    labels.extend([-100] * pad_len)
    attn.extend([0] * pad_len)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
        "question_input_ids": p_ids,
        "question_attention_mask": [1] * len(p_ids),
        "answers": a,
    }


def _collate(batch: List[Dict]):
    keys = ["input_ids", "labels", "attention_mask"]
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.stack([torch.tensor(b[k], dtype=torch.long) for b in batch])
    qi = [torch.tensor(b["question_input_ids"], dtype=torch.long) for b in batch]
    qa = [torch.tensor(b["question_attention_mask"], dtype=torch.long) for b in batch]
    out["question_input_ids"] = torch.nn.utils.rnn.pad_sequence(qi, batch_first=True, padding_value=0)
    out["question_attention_mask"] = torch.nn.utils.rnn.pad_sequence(qa, batch_first=True, padding_value=0)
    out["answers"] = [b["answers"] for b in batch]
    return out


def build_dataloaders(tokenizer, cfg: DictConfig):
    ds = datasets.load_dataset(cfg.dataset.name, cfg.dataset.config, cache_dir=CACHE_DIR)
    train_ds, dev_ds = ds[cfg.dataset.train_split], ds[cfg.dataset.eval_split]

    if cfg.mode == "trial":
        train_ds = train_ds.select(range(min(128, len(train_ds))))
        dev_ds = dev_ds.select(range(min(64, len(dev_ds))))

    fn = functools.partial(_proc, tokenizer=tokenizer, cfg=cfg)
    train_ds = train_ds.map(fn, remove_columns=train_ds.column_names, num_proc=cfg.dataset.num_workers)
    dev_ds = dev_ds.map(fn, remove_columns=dev_ds.column_names, num_proc=cfg.dataset.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.micro_batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        collate_fn=_collate,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.training.micro_batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=_collate,
    )
    return train_loader, dev_loader