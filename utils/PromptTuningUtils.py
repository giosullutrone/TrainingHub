from collections import Counter
from typing import List


def most_common_words(words: List[str], n: int):
    # Count occurrences of each string
    word_counts = Counter(words)

    # Get the N most common words
    most_common = word_counts.most_common(n)

    # Extract only the words from the result
    result = [item[0] for item in most_common]
    return result