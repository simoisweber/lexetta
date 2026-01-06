import torch
import time
from typing import Optional, Any, Tuple
from pathlib import Path
from transformers import Trainer
from datasets import DatasetDict

from schema import TrainingConfig, TrainingRun, TrainingTask, Metrics


def get_trainable_params(model: Any) -> tuple[int, int]:
    """
    Get trainable parameter statistics.
    
    Args:
        model: The model to analyze
        
    Returns:
        Tuple of (trainable params, total params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total



def extract_losses(trainer: Trainer) -> tuple[float, float]:
    """
    Extract final training and evaluation losses from trainer logs.
    
    Args:
        trainer: The Trainer instance after training
        
    Returns:
        Tuple of (final train loss, final eval loss)
    """
    logs = trainer.state.log_history
    final_train_loss = next(l["loss"] for l in reversed(logs) if "loss" in l)
    final_eval_loss = next(l["eval_loss"] for l in reversed(logs) if "eval_loss" in l)
    return final_train_loss, final_eval_loss


def save_results(
    data: TrainingRun,
    filepath: Path | str
) -> None:
    """
    Save training results to a JSON file.
    
    Args:
        data: The TrainingRun data to save
        filepath: Path to save the results
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w") as f:
        f.write(data.model_dump_json(indent=4))
    
    print(f"Results saved to {filepath}")

def train_model(trainer: Trainer) -> tuple[float, float]:
    """
    Train the model and collect timing/memory metrics.
    
    Args:
        trainer: The Trainer instance
        
    Returns:
        Tuple of (training time in seconds, peak VRAM in MB)
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    train_start = time.time()
    trainer.train()
    train_end = time.time()
    
    train_time = train_end - train_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    
    return train_time, peak_vram_mb



def run_single_training(
    config: TrainingConfig,
    dataset: DatasetDict,
    results_dir: str,
    results_filename: str,
    output_dir: str,
) -> TrainingRun:
    """
    Run a complete fine-tuning pipeline with a single configuration.
    
    Args:
        config: Training configuration
        dataset: Pre-loaded dataset
        output_dir: Directory for training outputs
        results_dir: Directory for saving results
        results_filename: Filename for results (auto-generated if None)
        max_length: Maximum sequence length for tokenization
        
    Returns:
        TrainingRun with all metrics
    """
    print(f"Starting training with config: {config}")
    
    if config.task == TrainingTask.CompLexV1:
        from lca.CompLex import tokenize_complex_dataset, create_trainer_complex, create_base_model, apply_lora
        
        # Tokenize
        print("Tokenizing dataset...")
        tokenized_dataset, tokenizer = tokenize_complex_dataset(dataset, max_length=config.max_input_length)
    
        # Create model
        print("Creating model with LoRA adapters...")
        model = create_base_model()
        model = apply_lora(model, config)
        trainable, total = get_trainable_params(model)
        
        # Train
        trainer = create_trainer_complex(
            model=model,
            config=config,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            output_dir=output_dir
        )
    else:
        raise NotImplementedError(f"Task {config.task} is unknown")
 
    print("Training...")    
    # evaluate once at the start for a baseline
    pre_train_eval = trainer.evaluate()
    # start the actual training
    train_time, peak_vram = train_model(trainer)
        
    # Extract metrics
    final_train_loss, final_eval_loss = extract_losses(trainer)

    # Create result object
    metrics = Metrics(
        train_time_s=train_time,
        peak_vram_mb=peak_vram,
        params_trainable=trainable,
        params_total=total,
        final_test_loss=final_eval_loss,
        final_train_loss=final_train_loss,
        logs=[pre_train_eval] + trainer.state.log_history
    )
    
    result = TrainingRun(
        config=config,
        metrics=metrics,
        version="1"
    )
    
    
    filepath = Path(results_dir) / results_filename
    save_results(result, filepath)
    
    return result


def run_batch_training(
    configs: list[TrainingConfig],
    dataset: DatasetDict,
    results_dir: str,
    results_filestem: str
) -> list[TrainingRun]:
    """
    Run multiple fine-tuning experiments with different configurations.
    
    Args:
        configs: List of TrainingConfig objects to run
        results_dir: Directory for saving results
        results_filestem: Result i is saved in i_<results_filestem>
    Returns:
        List of TrainingRun results
    """
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(configs)}")
        print(f"{'='*60}")
        
        result = run_single_training(
            config=config,
            dataset=dataset,
            output_dir=f"./lora-output",
            results_dir=results_dir,
            results_filename=f"{i+1}_{results_filestem}"
        )
        results.append(result)
    
    print("Batch training complete!")
    return results


def create_config_grid(
    ranks: list[int],
    alphas: list[int],
    target_modules: list[str],
    max_input_lengths: list[int], 
    learning_rates: list[float],
    batch_sizes: list[int],
    num_epochs: int,
    task: TrainingTask
) -> list[TrainingConfig]:
    """
    Create a grid of configurations for batch training.
    
    Args:
        ranks: List of LoRA ranks to try
        alphas: List of alpha values (defaults to 2x rank for each)
        learning_rates: List of learning rates to try
        batch_sizes: List of batch sizes to try
        num_epochs: Number of epochs for all configs
        target_modules: Which modules to apply LoRA to
        
    Returns:
        List of TrainingConfig objects
    """
    configs = []
    
    for rank in ranks:
        for alpha in alphas:
            for ml in max_input_lengths:
                for lr in learning_rates:
                    for bs in batch_sizes:
                        config = TrainingConfig(
                            task=task,
                            rank=rank,
                            alpha=alpha,
                            target_modules=target_modules,
                            max_input_length=ml,
                            learning_rate=lr,
                            batch_size=bs,
                            num_epochs=num_epochs,
                        )
                        configs.append(config)
        
    return configs