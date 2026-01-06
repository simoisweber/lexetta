from IPython.display import HTML, display

def get_color(score: float) -> str:
    """Convert 0-1 score to green-yellow-red gradient."""
    # score 0 = green, 0.5 = yellow, 1 = red
    if score < 0.5:
        r = int(255 * (score * 2))
        g = 255
    else:
        r = 255
        g = int(255 * (1 - (score - 0.5) * 2))
    return f"rgb({r}, {g}, 0)"

def visualize_complexity(sentence: str, word_scores: dict[str, float]) -> None:
    """
    Visualize word complexity in a sentence.
    
    Args:
        sentence: The input sentence
        word_scores: Dict mapping words to complexity scores (0-1)
    """
    html_parts = []
    for word in sentence.split():
        clean_word = word.strip(".,!?;:")
        score = word_scores.get(clean_word.lower(), 0.0)
        color = get_color(score)
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; '
            f'margin: 1px; border-radius: 3px;" title="{score:.2f}">{word}</span>'
        )
    
    html = f'<div style="font-size: 18px; line-height: 2;">{" ".join(html_parts)}</div>'
    display(HTML(html))

