# strategy_oracle.py

import numpy as np
from utils import save_strategy, load_strategy
from constants import HIDER_STRATEGY_FILE, STRATEGY_HISTORY_FILE
import random


class StrategyOracle:
    def __init__(self, game):
        self.game = game
        D = len(self.game.dictionary)
        self.hider_strategy = np.full(D, 1 / D)  # Uniform distribution
        self.iteration_history = []
        self.iteration_history.append(self.hider_strategy.copy())

    def update_hider_strategy(self, new_strategy):
        self.hider_strategy = new_strategy
        self.iteration_history.append(new_strategy.copy())

    def get_hider_best_response(self, avg_num_guesses):
        """
        Computes the hider's best response to the guesser's strategy.
        """
        max_guesses = np.max(avg_num_guesses)
        best_indices = np.where(avg_num_guesses == max_guesses)[0]
        best_response = np.zeros_like(self.hider_strategy)
        best_response[best_indices] = 1 / len(best_indices)
        return best_response

    def get_guesser_response(self, state):
        """
        Computes the guesser's move based on the current state and hider's strategy.
        """
        possible_indices = np.where(state)[0]
        expected_eliminations = []
        for guess_index in possible_indices:
            answer_probs = self.compute_answer_probs(guess_index, state)
            num_eliminated = self.compute_expected_eliminations(guess_index, state, answer_probs)
            expected_eliminations.append(num_eliminated)
        # Select the guess that maximizes expected eliminations
        best_guess_index = possible_indices[np.argmax(expected_eliminations)]
        return best_guess_index

    def compute_answer_probs(self, guess_index, state):
        """
        Computes the probability of each possible feedback (0 to word_length) for a given guess.
        """
        possible_indices = np.where(state)[0]
        common_letters = self.game.common_letters_matrix[guess_index, possible_indices]
        hider_probs = self.hider_strategy[possible_indices]
        answer_probs = np.zeros(self.game.word_length + 1)
        for feedback in range(self.game.word_length + 1):
            mask = common_letters == feedback
            prob = np.sum(hider_probs[mask])
            answer_probs[feedback] = prob
        # Normalize to sum to 1
        total_prob = np.sum(answer_probs)
        if total_prob > 0:
            answer_probs /= total_prob
        return answer_probs

    def compute_expected_eliminations(self, guess_index, state, answer_probs):
        """
        Computes the expected number of words eliminated by making a guess.
        """
        total_eliminations = 0
        possible_indices = np.where(state)[0]
        for feedback in range(self.game.word_length + 1):
            # For each possible feedback, calculate the number of words eliminated
            if answer_probs[feedback] == 0:
                continue
            words_eliminated = self.count_eliminated_words(guess_index, state, feedback)
            total_eliminations += answer_probs[feedback] * words_eliminated
        return total_eliminations

    def count_eliminated_words(self, guess_index, state, feedback):
        """
        Counts the number of words that would be eliminated given a feedback.
        """
        possible_indices = np.where(state)[0]
        common_letters = self.game.common_letters_matrix[guess_index, possible_indices]
        inconsistent = common_letters != feedback
        return np.sum(inconsistent)

    def sample_guesser_strategy(self, state):
        """
        Samples the guesser's strategy using the iteration history.
        """
        t = len(self.iteration_history)
        # Randomly select a past hider strategy
        i = random.randint(0, t - 1)
        hider_strategy = self.iteration_history[i]
        # Temporarily set hider strategy
        original_hider_strategy = self.hider_strategy.copy()
        self.hider_strategy = hider_strategy
        # Get the guess
        guess_index = self.get_guesser_response(state)
        # Restore the original hider strategy
        self.hider_strategy = original_hider_strategy
        return guess_index

    def save_hider_strategy(self, filename):
        save_strategy(self.hider_strategy, filename)

    def save_strategy_history(self, filename):
        strategy_array = np.array(self.iteration_history)
        np.save(filename, strategy_array)