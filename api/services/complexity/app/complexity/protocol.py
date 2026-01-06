from __future__ import annotations

import os
import logging
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LexicalComplexityScore(Protocol):
    """Protocol for Lexical Complexity Scoring systems
    """

    def score_words(self, sentence: str, words: List[str]) -> List[float]: 
        """Calculates the complexity score for each word in words
        
        Params:
            sentence: A string representing one sentence
            words: A list of words from the sentence
        
        Returns:
            The complexity score between 0.0 and 1.0 for each word in words

        """
    def score_sentence(self, sentence: str) -> float: 
        """Calculate the complexity score for a sentence
        
        Params:
            sentence: A string represening one sentence

        Returns:
            The complexity score between 0.0 and 1.0 for this sentence
        """
