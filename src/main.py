# src/main.py – orchestrator
from __future__ import annotations

import subprocess
import sys

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={str(cfg.results_dir)}",
        f"mode={cfg.mode}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()