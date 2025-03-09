import re
import random
from collections import Counter

def tokenize_text(file_path):
    """Reads a file, cleans text, and tokenizes it."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.split()

def generate_ngram_counts(tokens, n):
    """Generates n-gram and (n+1)-gram counts."""
    ngram_counts = Counter()
    n_plus1_gram_counts = Counter()
    
    for i in range(len(tokens) - n):
        ngram = tuple(tokens[i:i+n])
        n_plus1_gram = tuple(tokens[i:i+n+1])
        ngram_counts[ngram] += 1
        n_plus1_gram_counts[n_plus1_gram] += 1
    
    return ngram_counts, n_plus1_gram_counts

def count_ngram_matches(ngram_counts, n_plus1_gram_counts, input_text):
    """Finds matching (n+1)-grams and calculates probabilities."""
    words = tuple(input_text.lower().split())

    ngram_match_count = ngram_counts.get(words, 0)
    matching_n_plus1_grams = {}

    for n_plus1_gram, count in n_plus1_gram_counts.items():
        if n_plus1_gram[:len(words)] == words:
            probability = count / ngram_match_count if ngram_match_count > 0 else 0
            matching_n_plus1_grams[n_plus1_gram] = (count, probability)

    return ngram_match_count, matching_n_plus1_grams

def predict_next_word(matching_n_plus1_grams):
    """Predicts the next word based on highest probability."""
    if not matching_n_plus1_grams:
        return None  # No prediction possible

    max_prob = max(matching_n_plus1_grams.values(), key=lambda x: x[1])[1]
    top_candidates = [gram[-1] for gram, (count, prob) in matching_n_plus1_grams.items() if prob == max_prob]
    
    return random.choice(top_candidates)  # Pick randomly if multiple words have the same probability

# Example Usage
file_path = 'Mahabharata_kisari_mohan_ganguly.txt'
tokens = tokenize_text(file_path)

input_text = "the son of"
n = len(input_text.split())
ngram_counts, n_plus1_gram_counts = generate_ngram_counts(tokens, n)

match_count, matching_n_plus1_grams = count_ngram_matches(ngram_counts, n_plus1_gram_counts, input_text)

print(f"Exact N-Gram Match Count: {match_count}")
print(f"Matching N+1-Grams ({len(matching_n_plus1_grams)} found):")
for gram, (count, probability) in matching_n_plus1_grams.items():
    print(f"{' '.join(gram)} -> Count: {count}, Probability: {probability:.4f}")

predicted_word = predict_next_word(matching_n_plus1_grams)
print(f"Predicted Next Word: {predicted_word}")
