from __future__ import annotations
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List

from complexity.protocol import LexicalComplexityScore
from peft import PeftModel
import torch

class BertComplexityScorer(LexicalComplexityScore):
    """Fine-tuned BERT model with LoRA adapter for lexical complexity scoring."""
    
    def __init__(self, model_path: str, base_model: str = "bert-base-uncased", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer (this should work from your adapter directory)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model first
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=1,
        )
        
        # Load LoRA adapter on top
        self.model = PeftModel.from_pretrained(base, model_path).to(self.device)
        self.model.eval()   
    def _predict(self, sentence: str, word: str) -> float:
        """Run inference on a sentence-word pair."""

        inputs = self.tokenizer(
            sentence,
            word,
            return_tensors="pt",
            padding="max_length",
            truncation="only_first",
            max_length=128,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model(**inputs)
        
        return output.logits.squeeze().item()
    
    def score_sentence(self, sentence: str) -> float:
        """Average complexity of all words in the sentence."""
        words = sentence.split()
        if not words:
            return 0.0
        
        scores = self.score_words(sentence, words)
        return sum(scores) / len(scores)
    
    def score_words(self, sentence: str, words: List[str]) -> List[float]:
        return [self._predict(sentence, word) for word in words]