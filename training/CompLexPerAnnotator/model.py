
from typing import Any

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
from scipy import stats

from CompLexPerAnnotator.schema import TrainingConfig


def create_base_model(
    model_name: str = "bert-base-uncased",
) -> Any:
    """
    Create the base model.

    Params:
        model_name: Name of the pretrained model

    Returns:
        A tuple of the base model and the tokenizer
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer

def apply_lora(model: Any, config: TrainingConfig) -> Any:
    """
    Apply LoRA adapters to the model.

    Params:
        model: The base model
        config: Training configuration with LoRA parameters

    Returns:
        Model with LoRA adapters
    """
    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        use_rslora=True, # was proven to work better for different ranks https://doi.org/10.48550/arXiv.2312.03732
    )
    
    model = get_peft_model(model, lora_config)
    return model

def create_trainer_per_annotator(
    model: Any,
    config: TrainingConfig,
    train_dataset: Any,
    eval_dataset: Any,
    output_dir: str = None
) -> Trainer:
    """
    Create a Trainer instance.

    Params:
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
        pearson_r, _ = stats.pearsonr(predictions, labels)
        
        return {
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

def load_trained(model_dir) -> tuple:
    """
    Load a trained model and its tokenizer from a directory.

    Params:
        model_dir: Path to the saved PEFT model directory

    Returns:
        A tuple of the model and the tokenizer
    """
    base_model, tokenizer = create_base_model()
    model = PeftModel.from_pretrained(model=base_model, model_id=model_dir)
    return model, tokenizer

def predict_complexity(
    model,
    tokenizer,
    word: str,
    global_score: float,
    user_calibration: list[tuple[str, float, float]],
    n_calibration_examples: int = 5,
) -> float:
    """
    Predict the lexical complexity of a word for a specific user.

    Encoded as a BERT sequence pair to reflect information importance:
      - Segment A (most important): the target word and its global complexity score
        as a single bin token a-j, e.g. "profound e"
      - Segment B (less important): calibration words, each with their true user score
        and global predicted score as bin tokens, e.g. "ancient ef obscure cb ..."
        The contrast between true and predicted reveals the user's personal bias
        relative to the global model (see BiasFunction in data.py)

    Params:
        model: The trained model
        tokenizer: The tokenizer
        word: The target word to assess
        global_score: Complexity score predicted by the global (non-personalized) model
        user_calibration: Known (word, true_user_complexity, global_predicted_complexity)
                          triples for this user
        n_calibration_examples: How many calibration examples to include in segment B

    Returns:
        Predicted user complexity score as a float
    """
    def bin_token(score: float) -> str:
        return chr(ord('a') + min(int(score * 10), 9))

    score_token = bin_token(global_score)
    segment_a = f"{word} {score_token}"
    segment_b = " ".join(
        f"{w} {bin_token(true)}{bin_token(pred)}"
        for w, true, pred in user_calibration[:n_calibration_examples]
    )

    inputs = tokenizer(
        segment_a,
        segment_b,
        return_tensors="pt",
        padding="max_length",
        truncation="only_second",
        max_length=128,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    return output.logits.squeeze().item()