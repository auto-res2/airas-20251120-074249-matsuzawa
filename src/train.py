# src/train.py
"""Single-run executor with Hydra, WandB and Optuna (production-ready).
This version FIXES:
1.  Removed any local JSON metric files – *all* metrics live in WandB.
2.  Added mandatory energy ( ``joules_to_55_em`` ) and efficiency ( ``inference_flops`` )
    measurements and logging.
3.  Added missing dependencies (numpy, pandas, tqdm, pynvml) to *pyproject.toml*.
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.model import LoRALinear, build_model, get_lora_layers
from src.preprocess import build_dataloaders

# -----------------------------------------------------------------------------
#                               GPU ENERGY METER                               
# -----------------------------------------------------------------------------

class EnergyMeter:
    """Integrates GPU power draw over time using NVML.

    If NVML is unavailable the meter is *disabled* and returns ``nan``.
    """

    def __init__(self):
        try:
            import pynvml  # local import – listed in dependencies

            pynvml.nvmlInit()
            self._nvml = pynvml  # keep reference to prevent GC
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._last_t = time.time()
            self._energy = 0.0  # joules
            self.enabled = True
        except Exception:  # noqa: BLE001 – broad except deliberately
            self.enabled = False

    # ------------------------------------------------------------------
    def sample(self):
        if not self.enabled:
            return
        now = time.time()
        dt = now - self._last_t
        power_mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)  # milli-watts
        self._energy += (power_mw / 1000.0) * dt  # J = W * s
        self._last_t = now

    # ------------------------------------------------------------------
    def value(self) -> float:
        return float("nan") if not self.enabled else self._energy


# -----------------------------------------------------------------------------
#                               Repro & Safety                                  
# -----------------------------------------------------------------------------

def _set_seed(seed: int) -> None:  # deterministic-ish
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _assert_post_init(tokenizer, model, cfg: DictConfig) -> None:
    assert tokenizer.pad_token_id is not None, "[ASSERT] tokenizer.pad_token_id missing"
    dummy = torch.ones(1, 1, dtype=torch.long, device=model.device)
    logits = model(dummy).logits
    assert logits.shape[-1] == model.config.vocab_size, "[ASSERT] output dim ≠ vocab_size"
    assert cfg.model.lora.r0 > 0, "[ASSERT] initial LoRA rank must be positive"


def _assert_batch(batch) -> None:
    inp, lab = batch["input_ids"], batch["labels"]
    assert inp.shape == lab.shape, "[ASSERT] input/label shape mismatch"


def _assert_gradients(model: nn.Module) -> None:
    any_non_zero = False
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"[ASSERT] gradient of {n} is None"
        if not torch.allclose(p.grad, torch.zeros_like(p.grad)):
            any_non_zero = True
    assert any_non_zero, "[ASSERT] all retained gradients are zero – training stalled"


# -----------------------------------------------------------------------------
#                                  NACUS                                        
# -----------------------------------------------------------------------------

class NACUS:
    """Baseline: freezes LoRA params with consistently tiny gradients."""

    def __init__(self, gate: float, patience: int = 50):
        self.gate = gate
        self.patience = patience
        self._counters: Dict[str, int] = {}

    def apply(self, lora_layers: List[LoRALinear]):
        for layer in lora_layers:
            if all(not p.requires_grad for p in layer.params):
                continue
            max_abs = 0.0
            for p in layer.params:
                if p.grad is None:
                    continue
                max_abs = max(max_abs, p.grad.detach().abs().max().item())
            cnt = self._counters.get(layer.name, 0)
            cnt = cnt + 1 if max_abs < self.gate else 0
            self._counters[layer.name] = cnt
            if cnt >= self.patience:
                for p in layer.params:
                    p.requires_grad = False
                    p.grad = None


# -----------------------------------------------------------------------------
#                              GNARUM Controller                                
# -----------------------------------------------------------------------------

class GNARUMController:
    """Gradient-Noise Aware Rank & Update Modulator."""

    def __init__(self, cfg: DictConfig, layers: List[LoRALinear]):
        self.cfg, self.layers = cfg, layers
        self.state: Dict[str, Dict[str, float | int]] = {
            l.name: {"f": cfg.gnarum.f_init, "update_period": 1, "counter": 0, "r": l.r}
            for l in layers
        }

    # ------------------------------------------------------------------
    def before_step(self, step: int):
        for l in self.layers:
            upd_per = self.state[l.name]["update_period"]
            upd_now = (step % upd_per) == 0
            for p in l.params:
                p.requires_grad = upd_now

    # ------------------------------------------------------------------
    def after_update(
        self,
        step_lr: float,
        batch: int,
        grad_samples: Dict[str, List[torch.Tensor]],
    ) -> None:
        for l in self.layers:
            st = self.state[l.name]
            grads = grad_samples.get(l.name, [])
            if len(grads) < 2:
                continue
            g = torch.stack(grads)
            gns = g.std(0).mean().item()
            kappa = step_lr * math.sqrt(batch * st["f"]) / (gns * st["r"] + 1e-12)
            if kappa < self.cfg.gnarum.theta_low:
                st["counter"] += 1
                if st["counter"] >= self.cfg.gnarum.tau:
                    st["f"] = max(st["f"] / 2.0, self.cfg.gnarum.f_min)
                    st["update_period"] = int(round(1 / st["f"]))
                    st["counter"] = 0
            elif kappa > self.cfg.gnarum.theta_high:
                st["f"] = min(st["f"] * 2.0, 1.0)
                st["update_period"] = int(round(1 / st["f"]))
                st["counter"] = 0

    # ------------------------------------------------------------------
    def prune_and_regrow(self):
        for l in self.layers:
            st = self.state[l.name]
            r_now: int = int(st["r"])
            if r_now <= 1:
                continue
            keep = int(max(1, math.ceil(r_now * (1 - self.cfg.gnarum.p_prune))))
            with torch.no_grad():
                W = l.lora_B @ l.lora_A  # (out, in)
                try:
                    U, S, Vh = torch.linalg.svd(W.cpu(), full_matrices=False)
                except RuntimeError:  # numerical issues
                    continue
                idx = torch.argsort(S * torch.arange(1, S.numel() + 1), descending=True)[:keep]
                A_new = Vh[idx, :]
                B_new = (U[:, idx] * S[idx].unsqueeze(0))
                l.update_rank(keep, A_new.to(W.device), B_new.to(W.device))
                st["r"] = keep
            if (
                self.cfg.gnarum.regrow
                and st["r"] < self.cfg.model.lora.r0
                and st["f"] >= 1.0
            ):
                l.grow_rank(1)
                st["r"] = l.r


# -----------------------------------------------------------------------------
#                             Evaluation Utilities                              
# -----------------------------------------------------------------------------

def _strip(ans: str) -> str:
    ans = ans.strip()
    return ans[:-1] if ans.endswith(".") else ans


def evaluate(model, tokenizer, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    matches, total, losses = 0, 0, []
    with torch.no_grad():
        for batch in loader:
            q_ids = batch["question_input_ids"].to(model.device)
            q_att = batch["question_attention_mask"].to(model.device)
            answers: List[str] = batch["answers"]
            gen = model.generate(input_ids=q_ids, attention_mask=q_att, max_new_tokens=64)
            preds = tokenizer.batch_decode(gen[:, q_ids.size(1):], skip_special_tokens=True)
            for p, t in zip(preds, answers):
                matches += int(_strip(p) == _strip(t))
                total += 1
            inp_ids = batch["input_ids"].to(model.device)
            att = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            loss = model(input_ids=inp_ids, attention_mask=att, labels=labels).loss
            losses.append(loss.item())
    model.train()
    return matches / (total + 1e-9), float(np.mean(losses))


# -----------------------------------------------------------------------------
#                     FLOPs & parameter count estimation                        
# -----------------------------------------------------------------------------

def _infer_flops_and_params(layers: List[LoRALinear]) -> Tuple[int, int]:
    flops, params = 0, 0
    for l in layers:
        in_f, out_f, r = l.base.in_features, l.base.out_features, l.r
        # Forward pass: h→A (in_f*r) + (A h)→B (r*out_f)  multiply-acc = 2*MACs
        flops += 2 * (in_f * r + out_f * r)
        params += in_f * r + out_f * r
    return flops, params


# -----------------------------------------------------------------------------
#                             Training – single run                             
# -----------------------------------------------------------------------------

def _single_run(cfg: DictConfig, trial: Optional[optuna.Trial] = None) -> Tuple[float, Dict[str, float]]:
    _set_seed(cfg.training.seed)

    tokenizer, model = build_model(cfg)
    _assert_post_init(tokenizer, model, cfg)

    train_loader, dev_loader = build_dataloaders(tokenizer, cfg)
    lora_layers = get_lora_layers(model)

    gnarum = GNARUMController(cfg, lora_layers) if cfg.method == "gnarum" else None
    nacus = NACUS(cfg.model.nacus.gate_threshold) if cfg.method.startswith("lagnas_nacus") else None

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        betas=tuple(cfg.training.betas),
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.training.epochs * len(train_loader)
        )
        if cfg.training.scheduler == "cosine"
        else None
    )

    # WandB -------------------------------------------------------------
    use_wandb = trial is None and cfg.wandb.mode != "disabled"
    if use_wandb:
        import wandb

        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
        wandb.watch(model, log="all", log_freq=100)
    else:
        wandb = None  # type: ignore

    # Energy meter ------------------------------------------------------
    meter = EnergyMeter()
    energy_to_55_em: Optional[float] = None

    global_step, best_dev_em = 0, 0.0
    grad_storage: Dict[str, List[torch.Tensor]] = {}
    pbar = tqdm(total=cfg.training.epochs * len(train_loader), desc="train")

    for epoch in range(cfg.training.epochs):
        if gnarum and epoch % cfg.gnarum.svd_interval_epochs == 0 and epoch > 0:
            gnarum.prune_and_regrow()

        for step, batch in enumerate(train_loader):
            meter.sample()
            if global_step == 0:
                _assert_batch(batch)
            if gnarum:
                gnarum.before_step(global_step)

            batch = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss
            loss = loss / cfg.training.gradient_accumulation_steps
            loss.backward()

            # record gradients -------------------------------------------------
            if gnarum:
                for l in lora_layers:
                    for p in l.params:
                        if p.grad is not None:
                            grad_storage.setdefault(l.name, []).append(p.grad.detach().flatten())

            # optimisation ----------------------------------------------------
            if (global_step + 1) % cfg.training.gradient_accumulation_steps == 0:
                if nacus:
                    nacus.apply(lora_layers)
                _assert_gradients(model)
                optim.step()
                optim.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                if gnarum:
                    gnarum.after_update(
                        step_lr=optim.param_groups[0]["lr"],
                        batch=cfg.training.effective_batch_size,
                        grad_samples=grad_storage,
                    )
                grad_storage.clear()

            # -------- logging -------------------------------------------------
            if use_wandb:
                wandb.log({"train_loss": loss.item(), "lr": optim.param_groups[0]["lr"], "step": global_step})

            # -------- evaluation ---------------------------------------------
            if (
                (global_step + 1) % cfg.logging.eval_every_n_steps == 0
                or global_step == 0
            ):
                dev_em, vloss = evaluate(model, tokenizer, dev_loader)
                if dev_em > best_dev_em:
                    best_dev_em = dev_em
                if use_wandb:
                    wandb.log({"dev_exact_match": dev_em, "val_loss": vloss, "step": global_step})
                if best_dev_em >= 0.55 and energy_to_55_em is None:
                    energy_to_55_em = meter.value()
                if trial is not None:
                    trial.report(dev_em, global_step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            global_step += 1
            pbar.update(1)
            if cfg.mode == "trial" and global_step >= 2:
                break
        if cfg.mode == "trial":
            break
    pbar.close()

    # final eval --------------------------------------------------------
    final_em, final_vloss = evaluate(model, tokenizer, dev_loader)
    flops, params = _infer_flops_and_params(lora_layers)

    summary: Dict[str, float] = {
        "final_dev_exact_match": final_em,
        "final_val_loss": final_vloss,
        "joules_to_55_em": energy_to_55_em if energy_to_55_em is not None else float("nan"),
        "inference_flops": flops,
        "remaining_lora_params": params,
    }

    if use_wandb:
        for k, v in summary.items():
            wandb.summary[k] = v
        print("WandB URL:", wandb.run.get_url())
        wandb.finish()

    return final_em, summary


# -----------------------------------------------------------------------------
#                           Optuna hyper-search                                 
# -----------------------------------------------------------------------------

def _run_optuna(cfg: DictConfig) -> DictConfig:
    study = optuna.create_study(direction=cfg.optuna.direction)

    def objective(trial: optuna.Trial):
        cfg_t = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        for hp_path, spec in cfg.optuna.search_space.items():
            if spec["type"] == "loguniform":
                val = trial.suggest_float(hp_path, spec["low"], spec["high"], log=True)
            elif spec["type"] == "uniform":
                val = trial.suggest_float(hp_path, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                val = trial.suggest_categorical(hp_path, spec["choices"])
            else:
                raise ValueError(f"Unknown search space type {spec['type']}")
            OmegaConf.update(cfg_t, hp_path, val, merge=True)
        score, _ = _single_run(cfg_t, trial)
        return score

    study.optimize(objective, n_trials=cfg.optuna.n_trials)

    best_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for k, v in study.best_params.items():
        OmegaConf.update(best_cfg, k, v, merge=True)
    return best_cfg


# -----------------------------------------------------------------------------
#                                Hydra entry                                    
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: D401
    # merge run-specific YAML ---------------------------------------------------
    run_yaml = Path(get_original_cwd()) / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_yaml.exists():
        raise FileNotFoundError(f"Run-config not found: {run_yaml}")
    cfg_run = OmegaConf.load(run_yaml)
    cfg = OmegaConf.merge(cfg, cfg_run)

    # propagate run_id ---------------------------------------------------------
    cfg.run_id = cfg.get("run_id", cfg_run.run_id if "run_id" in cfg_run else cfg.run)

    # mode overrides -----------------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.training.epochs = 1
        cfg.optuna.n_trials = 0
        cfg.logging.eval_every_n_steps = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # optuna search ------------------------------------------------------------
    if cfg.optuna.n_trials and cfg.optuna.n_trials > 0:
        cfg = _run_optuna(cfg)

    _single_run(cfg)


if __name__ == "__main__":
    sys.exit(main())