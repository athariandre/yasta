#!/bin/bash
# Training launcher script for Tron PPO agent
# Ensures reproducibility with PYTHONHASHSEED and recommended defaults

set -e

# Set Python hash seed for reproducibility
export PYTHONHASHSEED=42

# Default parameters (can be overridden by environment variables)
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-200000}
ROLLOUT_LENGTH=${ROLLOUT_LENGTH:-2048}
BATCH_SIZE=${BATCH_SIZE:-256}
EPOCHS=${EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-0.0003}
DEVICE=${DEVICE:-cpu}
USE_FROZEN_OPPONENT=${USE_FROZEN_OPPONENT:-true}
OPPONENT_UPDATE_INTERVAL=${OPPONENT_UPDATE_INTERVAL:-5}
MAX_KL=${MAX_KL:-0.03}

# Create run directory if it doesn't exist
mkdir -p runs

# Echo configuration
echo "=================================================="
echo "Starting Tron PPO Training"
echo "=================================================="
echo "Seed: $SEED"
echo "Max steps: $MAX_STEPS"
echo "Rollout length: $ROLLOUT_LENGTH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Device: $DEVICE"
echo "Use frozen opponent: $USE_FROZEN_OPPONENT"
echo "Opponent update interval: $OPPONENT_UPDATE_INTERVAL"
echo "Max KL: $MAX_KL"
echo "=================================================="
echo ""

# Run training
python -m training.trainer \
    --seed $SEED \
    --max-steps $MAX_STEPS \
    --rollout-length $ROLLOUT_LENGTH \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --device $DEVICE \
    --use-frozen-opponent $USE_FROZEN_OPPONENT \
    --opponent-update-interval $OPPONENT_UPDATE_INTERVAL \
    --max-kl $MAX_KL \
    --csv-log true

echo ""
echo "=================================================="
echo "Training complete!"
echo "=================================================="
