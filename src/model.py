# src/model.py
"""Model builder and custom LoRA injection."""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = ".cache"

# -----------------------------------------------------------------------------
#                              LoRA module                                      
# -----------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Linear layer wrapped with LoRA adaptation."""

    def __init__(self, name: str, linear: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.name = name
        self.base = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(r, linear.in_features))
        self.lora_B = nn.Parameter(torch.empty(linear.out_features, r))
        self.reset_parameters()
        self.params = [self.lora_A, self.lora_B]

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        res = self.base(x)
        lora_out = torch.nn.functional.linear(self.dropout(x), self.lora_A)
        lora_out = torch.nn.functional.linear(lora_out, self.lora_B)
        return res + lora_out * self.scaling

    # rank adapt ---------------------------------------------------------------
    def update_rank(self, new_r: int, A_new: torch.Tensor, B_new: torch.Tensor):
        assert new_r < self.r, "new_r must be < current rank"
        device = self.lora_A.device
        self.lora_A = nn.Parameter(A_new.to(device).clone())
        self.lora_B = nn.Parameter(B_new.to(device).clone())
        self.r = new_r
        self.scaling = self.alpha / self.r
        self.params = [self.lora_A, self.lora_B]

    def grow_rank(self, add_r: int):
        if add_r <= 0:
            return
        device = self.lora_A.device
        extra_A = torch.randn(add_r, self.base.in_features, device=device) * 0.01
        extra_B = torch.zeros(self.base.out_features, add_r, device=device)
        self.lora_A = nn.Parameter(torch.cat([self.lora_A.data, extra_A], dim=0))
        self.lora_B = nn.Parameter(torch.cat([self.lora_B.data, extra_B], dim=1))
        self.r += add_r
        self.scaling = self.alpha / self.r
        self.params = [self.lora_A, self.lora_B]


# -----------------------------------------------------------------------------
#                         helper for module traversal                           
# -----------------------------------------------------------------------------

def _get_parent(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _inject_lora(model: nn.Module, cfg: DictConfig):
    targets = cfg.model.lora.target_modules
    r0, alpha, dropout = cfg.model.lora.r0, cfg.model.lora.alpha, cfg.model.lora.dropout
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if not any(t in name for t in targets):
            continue
        parent, child = _get_parent(model, name)
        setattr(parent, child, LoRALinear(name, mod, r0, alpha, dropout))


# -----------------------------------------------------------------------------
#                       public builder function                                 
# -----------------------------------------------------------------------------

def build_model(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model, cache_dir=CACHE_DIR, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if cfg.model.precision == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model, cache_dir=CACHE_DIR, torch_dtype=dtype, device_map="auto"
    )
    _inject_lora(model, cfg)
    return tokenizer, model


def get_lora_layers(model: nn.Module) -> List[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]