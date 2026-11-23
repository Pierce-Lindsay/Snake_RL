# Running the MLP DQN on WPI Turing

Training is CPU-bound (small network, one game step at a time), so these jobs
request CPUs only and use SLURM job arrays to run many configs in parallel.

## 1. Connect

On WPI WiFi or the GlobalProtect VPN:

```bash
ssh <wpi-username>@turing.wpi.edu
```

## 2. One-time setup (login node)

```bash
git clone -b mlp https://github.com/Pierce-Lindsay/Snake_RL.git
cd Snake_RL/snake_env
module load uv
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python numpy matplotlib \
    torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
mkdir -p runs/logs        # SLURM won't create the log directory itself
```

Quick sanity check (seconds, fine on the login node):

```bash
.venv/bin/python mlp.py -e 5 --eval 2 | grep -v "^Game over"
```

## 3. Hyperparameter sweep

```bash
sbatch cluster/sweep.sbatch       # 162 runs (54 configs x 3 seeds) packed into 4 jobs x 16 CPUs
squeue --me                       # monitor
```

When it finishes, rank configs by mean eval score across seeds:

```bash
grep -H EVAL runs/sweep/*/seed*/log.txt \
  | sed -E 's#runs/sweep/([^/]+)/.*mean=([0-9.]+).*#\1 \2#' \
  | awk '{s[$1]+=$2; n[$1]++} END {for (k in s) printf "%7.2f  %s (%d seeds)\n", s[k]/n[k], k, n[k]}' \
  | sort -rn | head
```

## 4. Final training

Put the winning hyperparameters in `HPARAMS` in `cluster/train.sbatch`, then:

```bash
sbatch cluster/train.sbatch       # 3 seeds x 3000 episodes
grep EVAL runs/final/seed*/log.txt
```

Each run directory has `mlp_snake.pt`, `training_progress_mlp.png` and `log.txt`.
Copy results back to your machine with:

```bash
scp -r <wpi-username>@turing.wpi.edu:Snake_RL/snake_env/runs/final ./
```

## Notes

- Jobs use the `academic` partition: course accounts (cs551) are denied on `short`. Limits:
  64 CPUs and 24 running jobs per user, 2-day wall time.
- Useful commands: `squeue --me`, `sacct`, `seff <jobid>`, `scancel <jobid>`.
- Docs: https://docs.turing.wpi.edu/
