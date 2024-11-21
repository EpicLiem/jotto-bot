# utils.py

import numpy as np
import os
from constants import WORD_LENGTH, DICTIONARY_FILE, COMMON_LETTERS_FILE

def load_dictionary(word_length):
    """
    Loads and preprocesses the dictionary.
    Removes words with duplicate letters and words that are anagrams of other words.
    """
    dictionary = []
    words_seen = set()
    anagrams_seen = set()
    with open(DICTIONARY_FILE, 'r') as f:
        for line in f:
            word = line.strip().lower()
            if len(word) != word_length:
                continue
            if len(set(word)) != word_length:
                continue  # Skip words with duplicate letters
            word_sorted = ''.join(sorted(word))
            if word_sorted in anagrams_seen:
                continue  # Skip words that are anagrams of already seen words
            anagrams_seen.add(word_sorted)
            if word not in words_seen:
                dictionary.append(word)
                words_seen.add(word)
    return dictionary

def precompute_common_letters(dictionary):
    """
    Precomputes the common letters matrix and saves it to a file.
    """
    D = len(dictionary)
    common_letters = np.zeros((D, D), dtype=int)
    for i in range(D):
        word_i = set(dictionary[i])
        for j in range(i, D):
            word_j = set(dictionary[j])
            common = len(word_i & word_j)
            common_letters[i, j] = common
            common_letters[j, i] = common  # Symmetric
    np.save(COMMON_LETTERS_FILE, common_letters)

def load_common_letters():
    """
    Loads the precomputed common letters matrix.
    """
    if not os.path.exists(COMMON_LETTERS_FILE):
        raise FileNotFoundError("Common letters file not found. Run precompute_common_letters first.")
    return np.load(COMMON_LETTERS_FILE)

def save_strategy(strategy, filename):
    """
    Saves a strategy vector to a file.
    """
    np.save(filename, strategy)

def load_strategy(filename):
    """
    Loads a strategy vector from a file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Strategy file {filename} not found.")
    return np.load(filename)

def parallel_compute(function, iterable, num_workers):
    """
    Helper for parallel computations.
    """
    from multiprocessing import Pool
    with Pool(processes=num_workers) as pool:
        results = pool.map(function, iterable)
    return results