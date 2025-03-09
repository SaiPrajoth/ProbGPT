# ProbGPT: N-Gram Based Next Word Prediction

## Introduction
ProbGPT is a statistical language model that predicts the most probable next word given a sequence of words using **pure probability-based N-gram modeling**. This model does not use deep learning, machine learning, or any advanced NLP techniques—it is purely built on **word frequency probabilities** extracted from the dataset.

## Methodology
The model is based on **N-Gram Language Modeling**, where:
- An **N-gram** is a sequence of **N words** appearing consecutively in a given corpus.
- The probability of the next word is computed using **conditional probability**:

$$
P(w_{N+1} \mid w_1, w_2, \dots, w_N) = \frac{C(w_1, w_2, \dots, w_{N+1})}{C(w_1, w_2, \dots, w_N)}
$$

where:
- $C(w_1, w_2, \dots, w_N)$ is the count of the **N-gram** in the dataset.
- $C(w_1, w_2, \dots, w_{N+1})$ is the count of the **(N+1)-gram**.

This model **solely relies on probability calculations** and does not incorporate context understanding beyond N-gram frequency.

## Key Components

### **1. Tokenization**
The text is preprocessed by:
- Converting to lowercase.
- Removing numbers and punctuation.
- Splitting into tokens (words).

### **2. N-Gram Model Construction**
Two frequency distributions are generated:
- **N-gram count dictionary** $C(w_1, w_2, \dots, w_N)$
- **(N+1)-gram count dictionary** $C(w_1, w_2, \dots, w_{N+1})$

### **3. Next Word Prediction**
- For a given input sequence, all possible **(N+1)-grams** that start with the input are found.
- The next word is chosen as the most frequent one based on probability.
- If multiple candidates have the same frequency, one is selected randomly.

## Implementation
### **Generating N-Grams**
```python
from collections import Counter

def generate_ngram_counts(tokens, n):
    ngram_counts = Counter()
    n_plus1_gram_counts = Counter()
    
    for i in range(len(tokens) - n):
        ngram = tuple(tokens[i:i+n])
        n_plus1_gram = tuple(tokens[i:i+n+1])
        ngram_counts[ngram] += 1
        n_plus1_gram_counts[n_plus1_gram] += 1
    
    return ngram_counts, n_plus1_gram_counts
