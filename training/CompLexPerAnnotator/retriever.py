from typing import Protocol
import numpy as np
from wordfreq import zipf_frequency

class Retriever(Protocol):
    def __call__(self, sample: dict, n: int) -> list:
        """
        Retrieves n items from the history based on *sample*

        Params:
            sample: a row dict (keys: task_id, annotator_id, corpus, sentence, token, complexity)
            n: number of items to return

        Returns:
            Returns a list of row dicts of length n
        """

class RandomRetriever:
    def __init__(self, history: list, seed: int = None):
        self.history = history
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: dict, n: int) -> list:
        indices = self.rng.choice(len(self.history), size=min(n, len(self.history)), replace=False)
        return [self.history[i] for i in indices]


class WordFrequencyRetriever:
    def __init__(self, history: list, lang: str = "en"):
        self.history = history
        self.lang = lang
        self._freqs = [zipf_frequency(item["token"], lang) for item in history]

    def __call__(self, sample: dict, n: int) -> list:
        query_freq = zipf_frequency(sample["token"], self.lang)
        ranked = sorted(
            range(len(self.history)),
            key=lambda i: abs(self._freqs[i] - query_freq),
        )
        return [self.history[i] for i in ranked[:min(n, len(self.history))]]
