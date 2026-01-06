from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from complexity.protocol import LexicalComplexityScore
from complexity.bert import BertComplexityScorer
from constants import MODEL_DIR

# ============================================================================
# Request/Response Models
# ============================================================================

# --- Single Requests ---

class SentenceRequest(BaseModel):
    sentence: str = Field(..., min_length=1, description="The sentence to analyze")

class WordRequest(BaseModel):
    sentence: str = Field(..., min_length=1, description="The sentence containing the words")
    words: List[str] = Field(..., min_length=1, description="Words to analyze from the sentence")

# --- Batch Requests ---

class BatchSentenceRequest(BaseModel):
    sentences: List[str] = Field(..., min_length=1, description="List of sentences to analyze")

class BatchWordRequest(BaseModel):
    requests: List[WordRequest] = Field(..., min_length=1, description="List of word analysis requests")

# --- Responses ---

class SentenceResponse(BaseModel):
    sentence: str
    complexity: float = Field(..., ge=0.0, le=1.0, description="Complexity score between 0 and 1")

class WordScore(BaseModel):
    word: str
    complexity: float = Field(..., ge=0.0, le=1.0)

class WordResponse(BaseModel):
    sentence: str
    scores: List[WordScore]

class BatchSentenceResponse(BaseModel):
    results: List[SentenceResponse]

class BatchWordResponse(BaseModel):
    results: List[WordResponse]


# ============================================================================
# Application
# ============================================================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="Lexical Complexity API",
        version="1.0.0",
        description=(
            "API for predicting lexical complexity for words in a sentence.\n\n"
            "**Endpoints:**\n"
            "- `POST /v1/complex/sentence` - analyze full sentence\n"
            "- `POST /v1/complex/word` - analyze word(s) in a sentence\n"
            "- `POST /v1/complex/batch/sentence` - batch sentence analysis\n"
            "- `POST /v1/complex/batch/word` - batch word analysis\n"
        ),
    )
    return app


app = create_app()

scorer: Optional[LexicalComplexityScore] = BertComplexityScorer(model_path=MODEL_DIR / "bert")
def get_scorer() -> LexicalComplexityScore:
    if scorer is None:
        raise HTTPException(status_code=503, detail="Scorer not initialized")
    return scorer


# ============================================================================
# Endpoints
# ============================================================================

@app.post(
    "/v1/complex/sentence",
    response_model=SentenceResponse,
    summary="Analyze sentence complexity",
    tags=["Single"],
)
def analyze_sentence(request: SentenceRequest) -> SentenceResponse:
    """Calculate the complexity score for an entire sentence."""
    s = get_scorer()
    complexity = s.score_sentence(request.sentence)
    return SentenceResponse(sentence=request.sentence, complexity=complexity)


@app.post(
    "/v1/complex/word",
    response_model=WordResponse,
    summary="Analyze word complexity",
    tags=["Single"],
)
def analyze_words(request: WordRequest) -> WordResponse:
    """Calculate complexity scores for specific words in a sentence."""
    s = get_scorer()
    scores = s.score_words(request.sentence, request.words)
    
    word_scores = [
        WordScore(word=word, complexity=score)
        for word, score in zip(request.words, scores)
    ]
    return WordResponse(sentence=request.sentence, scores=word_scores)


@app.post(
    "/v1/complex/batch/sentence",
    response_model=BatchSentenceResponse,
    summary="Batch analyze sentences",
    tags=["Batch"],
)
def analyze_sentences_batch(request: BatchSentenceRequest) -> BatchSentenceResponse:
    """Calculate complexity scores for multiple sentences."""
    s = get_scorer()
    results = [
        SentenceResponse(sentence=sentence, complexity=s.score_sentence(sentence))
        for sentence in request.sentences
    ]
    return BatchSentenceResponse(results=results)


@app.post(
    "/v1/complex/batch/word",
    response_model=BatchWordResponse,
    summary="Batch analyze words",
    tags=["Batch"],
)
def analyze_words_batch(request: BatchWordRequest) -> BatchWordResponse:
    """Calculate complexity scores for words across multiple sentences."""
    s = get_scorer()
    results = []
    
    for req in request.requests:
        scores = s.score_words(req.sentence, req.words)
        word_scores = [
            WordScore(word=word, complexity=score)
            for word, score in zip(req.words, scores)
        ]
        results.append(WordResponse(sentence=req.sentence, scores=word_scores))
    
    return BatchWordResponse(results=results)
