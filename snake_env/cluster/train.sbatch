#!/bin/bash
# Final training run with the chosen hyperparameters, 3 seeds in parallel in one job.
# Edit the hyperparameters below to the sweep winner, then from snake_env/:
#   sbatch cluster/train.sbatch
# Extended 14-input state, saved to runs/extended/ instead of runs/final/:
#   sbatch --export=ALL,RUN_NAME=extended,EXTRA_ARGS=--extended-state cluster/train.sbatch
#SBATCH --job-name=snake-mlp-train
#SBATCH --partition=academic        # course account cs551 is denied on short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6g
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/logs/train_%j.out

EPISODES=3000
HPARAMS="--lr 5e-4 --gamma 0.9 --batch-size 256 --dropout 0.1 $EXTRA_ARGS"
export OMP_NUM_THREADS=1

for seed in 0 1 2; do
    (
        dir="$SLURM_SUBMIT_DIR/runs/${RUN_NAME:-final}/seed${seed}"
        mkdir -p "$dir" && cd "$dir"
        "$SLURM_SUBMIT_DIR/.venv/bin/python" -u "$SLURM_SUBMIT_DIR/mlp.py" \
            -e $EPISODES --eval 100 --seed $seed $HPARAMS \
            | grep -v "^Game over" > log.txt
        echo "seed${seed}: $(grep EVAL log.txt)"
    ) &
done
wait
