import sys
import json
from pathlib import Path

from CompLexPerAnnotator import load_dataset, preprocess_data, run_single_training, save_results, TrainingConfig


if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <config.json> <output.json>")
    sys.exit(1)

config_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

with open(config_path) as f:
    config = TrainingConfig.model_validate(json.load(f))

data = load_dataset()
data = preprocess_data(data)

trainer, run = run_single_training(
    config=config,
    dataset=data,
    output_dir=None
)

print(f"Training time:    {run.metrics.train_time_s:.1f}s")
print(f"Peak VRAM:        {run.metrics.peak_vram_mb:.1f} MB")
print(f"Trainable params: {run.metrics.params_trainable:,} / {run.metrics.params_total:,}")
print(f"Train loss:       {run.metrics.final_train_loss:.4f}")
print(f"Test loss:        {run.metrics.final_test_loss:.4f}")

save_results(run, output_path)
