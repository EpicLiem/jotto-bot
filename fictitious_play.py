# fictitious_play.py

import numpy as np
from strategy_oracle import StrategyOracle
from jotto_game import JottoGame
from constants import NUM_ITERATIONS, HIDER_STRATEGY_FILE, STRATEGY_HISTORY_FILE
from multiprocessing import Pool
import tqdm

def compute_num_guesses_for_word(game, oracle, word_index):
    """
    Computes the number of guesses required for the guesser's strategy to guess a specific word.
    """
    game_copy = JottoGame(game.word_length)
    game_copy.dictionary = game.dictionary
    game_copy.common_letters_matrix = game.common_letters_matrix
    game_copy.reset_state()
    num_guesses = 0
    state = game_copy.state.copy()
    while True:
        num_guesses += 1
        guess_index = oracle.get_guesser_response(state)
        feedback = game_copy.num_common_letters(guess_index, word_index)
        if feedback == game_copy.word_length:
            break
        game_copy.update_state(guess_index, feedback)
        state = game_copy.state.copy()
    return num_guesses

class FictitiousPlay:
    def __init__(self, game, oracle, num_iterations=NUM_ITERATIONS):
        self.game = game
        self.oracle = oracle
        self.num_iterations = num_iterations
        self.epsilon_history = []
        self.best_epsilon = float('inf')
        self.best_iteration = 0
        self.avg_num_guesses = None  # Will be initialized in run()
        self.iteration = 0

    def run(self):
        D = len(self.game.dictionary)
        # Initialize avg_num_guesses and ing (iteration num guesses)
        self.avg_num_guesses = np.zeros(D)
        self.ing = np.zeros(D)

        # Precompute indices for parallelization
        word_indices = np.arange(D)

        for t in tqdm.tqdm(range(1, self.num_iterations + 1)):
            self.iteration = t
            # Compute the guesser's response to the current hider strategy
            self.ing = self.compute_num_guesses_parallel()
            # Update avg_num_guesses
            self.avg_num_guesses = ((t - 1) / t) * self.avg_num_guesses + (1 / t) * self.ing
            # Compute hider's best response
            hider_best_response = self.oracle.get_hider_best_response(self.avg_num_guesses)
            # Update hider's strategy
            self.oracle.hider_strategy = ((t - 1) / t) * self.oracle.hider_strategy + (1 / t) * hider_best_response
            self.oracle.update_hider_strategy(self.oracle.hider_strategy)
            # Compute epsilon
            epsilon = self.compute_epsilon()
            self.epsilon_history.append(epsilon)
            if epsilon < self.best_epsilon:
                self.best_epsilon = epsilon
                self.best_iteration = t
            # Optionally, print progress
            # print(f"Iteration {t}, Epsilon: {epsilon:.4f}, Best Epsilon: {self.best_epsilon:.4f}")

        # Save the final hider strategy and strategy history
        self.oracle.save_hider_strategy(HIDER_STRATEGY_FILE)
        self.oracle.save_strategy_history(STRATEGY_HISTORY_FILE)
        print(f"Best Epsilon: {self.best_epsilon:.4f} at iteration {self.best_iteration}")

    def compute_num_guesses_parallel(self):
        D = len(self.game.dictionary)
        word_indices = np.arange(D)
        num_workers = None  # Use all available processors
        with Pool(processes=num_workers) as pool:
            ing = pool.map(self.compute_num_guesses_for_word_wrapper, word_indices)
        return np.array(ing)

    def compute_num_guesses_for_word_wrapper(self, word_index):
        """
        Wrapper function for multiprocessing; required because methods cannot be directly pickled.
        """
        return compute_num_guesses_for_word(self.game, self.oracle, word_index)

    def compute_epsilon(self):
        """
        Computes the exploitability (epsilon) for both players.
        """
        hider_payoff_actual = np.dot(self.avg_num_guesses, self.oracle.hider_strategy)
        hider_payoff_best = np.max(self.avg_num_guesses)
        hider_epsilon = hider_payoff_best - hider_payoff_actual

        guesser_payoff_actual = -hider_payoff_actual
        guesser_payoff_best = -np.min(self.avg_num_guesses)
        guesser_epsilon = guesser_payoff_best - guesser_payoff_actual

        epsilon = max(hider_epsilon, guesser_epsilon)
        return epsilon