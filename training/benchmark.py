import json
import argparse
from pathlib import Path

import torch
from pydantic import ValidationError
from scipy import stats

from CompLexPerAnnotator.schema import TrainingConfig as PAConfig
from CompLexPerAnnotator.data import load_dataset, preprocess_data, get_user_histories, tokenize_per_annotator_dataset
from CompLexPerAnnotator.model import load_trained as pa_load_trained
from CompLexPerAnnotator.train import get_retriever
from CompLex.schema import TrainingConfig as ComplexConfig
from CompLex.model import load_trained as complex_load_trained


def detect_type(config_path: Path):
    with open(config_path) as f:
        raw = json.load(f)
    try:
        config = PAConfig.model_validate(raw)
        return "per_annotator", config
    except ValidationError:
        config = ComplexConfig.model_validate(raw)
        return "complex", config


def tokenize_for_complex(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            batch["token"],
            padding="max_length",
            truncation="only_first",
            max_length=128,
            return_token_type_ids=True,
        )

    test = dataset["test"].map(tokenize, batched=True)
    test = test.remove_columns(["task_id", "annotator_id", "corpus", "sentence", "token"])
    test = test.rename_column("complexity", "labels")
    test.set_format("torch")
    return test


def run_inference(model, test_data, batch_size=16) -> float:
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_labels = []

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        all_preds.extend(output.logits.flatten().cpu().tolist())
        all_labels.extend(labels.tolist())

    pearson_r, _ = stats.pearsonr(all_preds, all_labels)
    return float(pearson_r)


def main():
    parser = argparse.ArgumentParser(description="Benchmark trained models on CompLexPerAnnotator test set")
    parser.add_argument("paths", nargs="+", help="Run directories to evaluate")
    parser.add_argument("-o", "--output", help="Save results to JSON file")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_dataset()
    dataset = preprocess_data(dataset)
    user_histories = get_user_histories(dataset)

    results = []

    for path_str in args.paths:
        path = Path(path_str)
        model_type, config = detect_type(path / "config.json")
        print(f"\nEvaluating {path_str} ({model_type})...")

        if model_type == "per_annotator":
            model, tokenizer = pa_load_trained(str(path / "model"))
            retriever_map = {
                aid: get_retriever(retriever_type=config.retriever_type, history=history)
                for aid, history in user_histories.items()
            }
            tokenized = tokenize_per_annotator_dataset(
                dataset,
                tokenizer=tokenizer,
                retriever_map=retriever_map,
                user_history_length=config.user_history_length,
            )
            test_data = tokenized["test"]
        else:
            model, tokenizer = complex_load_trained(str(path / "model"))
            test_data = tokenize_for_complex(dataset, tokenizer)

        if torch.cuda.is_available():
            model = model.cuda()

        pearson_r = run_inference(model, test_data)
        results.append({"path": path_str, "pearson_r": pearson_r})

    print()
    for r in results:
        print(f"{r['path']:<50} {r['pearson_r']:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
