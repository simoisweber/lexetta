from typing import Protocol
import numpy as np

class Retriever(Protocol):
    def __call__(self, sample: tuple, n: int) -> list:
        """
        Retrieves n items from the history based on *sample*

        Params:
            sample: a tuple of (context: str, token: str)
            n: number of items to return
        
        Returns:
            Returns a list of tuples (sample, score) of length n 
        """

class RandomRetriever:
    def __init__(self, history: list, seed: int = None):
        self.history = history
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: tuple, n: int):
        indices = self.rng.choice(len(self.history), size=min(n, len(self.history)), replace=False)
        return [self.history[i] for i in indices]
