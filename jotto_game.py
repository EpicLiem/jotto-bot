# jotto_game.py

import numpy as np
from utils import load_dictionary, load_common_letters
from constants import WORD_LENGTH


class JottoGame:
    def __init__(self, word_length):
        self.word_length = word_length
        self.dictionary = load_dictionary(word_length)
        self.common_letters_matrix = load_common_letters()
        self.reset_state()

    def reset_state(self):
        self.state = np.ones(len(self.dictionary), dtype=bool)

    def update_state(self, guess_index, feedback):
        """
        Updates the game state based on the guess and feedback.
        Eliminates words inconsistent with the feedback.
        """
        possible_indices = np.where(self.state)[0]
        # Get the number of common letters between the guess and all possible words
        common_letters = self.common_letters_matrix[guess_index, possible_indices]
        # Keep words where the number of common letters matches the feedback
        consistent = common_letters == feedback
        # Update the state
        self.state[possible_indices] = consistent

    def num_common_letters(self, index1, index2):
        """
        Returns the number of common letters between two words using precomputed matrix.
        """
        return self.common_letters_matrix[index1, index2]

    def get_possible_words(self):
        """
        Returns the indices of words that are still possible.
        """
        return np.where(self.state)[0]

    def get_dictionary(self):
        """
        Returns the dictionary.
        """
        return self.dictionary