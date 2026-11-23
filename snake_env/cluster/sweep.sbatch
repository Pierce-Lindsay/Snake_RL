#!/bin/bash
# Hyperparameter grid search for the MLP DQN.
# Grid: lr(3) x gamma(3) x batch(2) x dropout(3) = 54 configs x 3 seeds = 162 runs.
# The runs are split across a small job array (to stay under the per-user job
# limit); each job runs its share in parallel, one run per CPU.
# Submit from snake_env/:  sbatch cluster/sweep.sbatch
#SBATCH --job-name=snake-mlp-sweep
#SBATCH --partition=academic        # course account cs551 is denied on short; academic: 64 CPUs/user, 2 days
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16          # runs executed in parallel per job
#SBATCH --mem=16g
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3                 # 4 jobs x 16 CPUs = 64 (academic per-user cap)
#SBATCH --output=runs/logs/sweep_%A_%a.out

export EPISODES=1000 SUBMIT_DIR=$SLURM_SUBMIT_DIR OMP_NUM_THREADS=1
TOTAL_RUNS=162

run_one() {
    local LRS=(5e-4 1e-3 3e-3) GAMMAS=(0.9 0.95 0.99) BATCHES=(64 256) DROPOUTS=(0.0 0.1 0.2)
    local i=$1
    local seed=$((i % 3));                   i=$((i / 3))
    local dropout=${DROPOUTS[$((i % 3))]};   i=$((i / 3))
    local batch=${BATCHES[$((i % 2))]};      i=$((i / 2))
    local gamma=${GAMMAS[$((i % 3))]};       i=$((i / 3))
    local lr=${LRS[$i]}
    local dir="$SUBMIT_DIR/runs/sweep/lr${lr}_g${gamma}_b${batch}_d${dropout}/seed${seed}"
    mkdir -p "$dir" && cd "$dir" || return
    # snake.py prints every death; drop those lines to keep logs small
    "$SUBMIT_DIR/.venv/bin/python" -u "$SUBMIT_DIR/mlp.py" \
        -e $EPISODES --eval 100 --seed $seed \
        --lr $lr --gamma $gamma --batch-size $batch --dropout $dropout \
        | grep -v "^Game over" > log.txt
    echo "$dir: $(grep EVAL log.txt)"
}
export -f run_one

# This job takes runs task_id, task_id + N, task_id + 2N, ...
seq $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $((TOTAL_RUNS - 1)) \
    | xargs -P $SLURM_CPUS_PER_TASK -I{} bash -c 'run_one {}'
