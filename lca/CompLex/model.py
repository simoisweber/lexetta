
from typing import Any

from transformers import AutoModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from scipy import stats

from lca.schema import TrainingConfig


def create_base_model(
    model_name: str = "bert-base-uncased",
    freeze_backbone: bool = True
) -> Any:
    """
    Create and optionally freeze the base model.
    
    Args:
        model_name: Name of the pretrained model
        freeze_backbone: Whether to freeze backbone parameters
        
    Returns:
        The base model
    """
    model = AutoModel.from_pretrained(
        model_name,
    )

    if freeze_backbone:
        # freeze ONLY the encoder/backbone
        for p in model.bert.parameters():
            p.requires_grad = False

        # ensure the QA head stays trainable
        for p in model.qa_outputs.parameters():
            p.requires_grad = True
           
    return model

def apply_lora(model: Any, config: TrainingConfig) -> Any:
    """
    Apply LoRA adapters to the model.
    
    Args:
        model: The base model
        config: Training configuration with LoRA parameters
        lora_dropout: Dropout rate for LoRA layers
        
    Returns:
        Model with LoRA adapters
    """
    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        use_rslora=True # was proven to work better for different ranks https://doi.org/10.48550/arXiv.2312.03732
    )
    
    model = get_peft_model(model, lora_config)
    return model

def create_trainer_complex(
    model: Any,
    config: TrainingConfig,
    train_dataset: Any,
    eval_dataset: Any,
    output_dir: str = "./outputs"
) -> Trainer:
    """
    Create a Trainer instance.
    
    Args:
        model: The model to train
        config: Training configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Directory for outputs
        
    Returns:
        Configured Trainer instance
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze()  # (batch, 1) -> (batch,)
        
        mse = ((predictions - labels) ** 2).mean()
        mae = np.abs(predictions - labels).mean()
        pearson_r, _ = stats.pearsonr(predictions, labels)
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "pearson_r": float(pearson_r),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        logging_steps=100,
    )
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
