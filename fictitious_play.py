# fictitious_play.py
import os

import numpy as np
from strategy_oracle import StrategyOracle
from jotto_game import JottoGame
from constants import NUM_ITERATIONS, HIDER_STRATEGY_FILE, STRATEGY_HISTORY_FILE, CHECKPOINT_DIR, S3_BUCKET_NAME
from multiprocessing import Pool
import tqdm
import boto3

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
    def __init__(self, game, oracle, num_iterations=5000, checkpoint_dir="checkpoints"):
        self.game = game
        self.oracle = oracle
        self.num_iterations = num_iterations
        self.checkpoint_dir = checkpoint_dir
        self.epsilon_history = []
        self.best_epsilon = float('inf')
        self.best_iteration = 0
        self.avg_num_guesses = None
        self.iteration = 0

        # Ensure checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def run(self):
        """Runs the Fictitious Play algorithm with tqdm for progress visualization."""
        self.load_checkpoint()  # Try to resume from a checkpoint
        D = len(self.game.dictionary)

        if self.avg_num_guesses is None:
            self.avg_num_guesses = np.zeros(D)
            for t in tqdm.tqdm(range(self.iteration + 1, self.num_iterations + 1)):
                self.iteration = t

                # Compute the guesser's response to the current hider strategy
                ing = self.compute_num_guesses_parallel()
                self.avg_num_guesses = ((t - 1) / t) * self.avg_num_guesses + (1 / t) * ing

                # Compute hider's best response
                hider_best_response = self.oracle.get_hider_best_response(self.avg_num_guesses)
                self.oracle.hider_strategy = ((t - 1) / t) * self.oracle.hider_strategy + (1 / t) * hider_best_response
                self.oracle.update_hider_strategy(self.oracle.hider_strategy)

                # Compute epsilon
                epsilon = self.compute_epsilon()
                self.epsilon_history.append(epsilon)

                if epsilon < self.best_epsilon:
                    self.best_epsilon = epsilon
                    self.best_iteration = t

                # Save checkpoints every 100 iterations
                if t % 100 == 0:
                    self.save_checkpoint()

            # Save the final results
            self.save_checkpoint()
            print(f"Best Epsilon: {self.best_epsilon:.4f} at iteration {self.best_iteration}")

    def save_checkpoint(self):
        """Saves the current state of the training process."""
        np.save(os.path.join(self.checkpoint_dir, "hider_strategy.npy"), self.oracle.hider_strategy)
        np.save(os.path.join(self.checkpoint_dir, "strategy_history.npy"), np.array(self.oracle.iteration_history))
        np.save(os.path.join(self.checkpoint_dir, "avg_num_guesses.npy"), self.avg_num_guesses)
        np.save(os.path.join(self.checkpoint_dir, "epsilon_history.npy"), np.array(self.epsilon_history))
        with open(os.path.join(self.checkpoint_dir, "iteration.txt"), "w") as f:
            f.write(str(self.iteration))
        print(f"Checkpoint saved at iteration {self.iteration}")

        # Upload checkpoint to S3 if configured
        if S3_BUCKET_NAME:
            self.upload_to_s3()

    def load_checkpoint(self):
        """Loads the last saved state of the training process."""
        try:
            self.oracle.hider_strategy = np.load(os.path.join(self.checkpoint_dir, "hider_strategy.npy"))
            self.oracle.iteration_history = list(
                np.load(os.path.join(self.checkpoint_dir, "strategy_history.npy"), allow_pickle=True))
            self.avg_num_guesses = np.load(os.path.join(self.checkpoint_dir, "avg_num_guesses.npy"))
            self.epsilon_history = list(
                np.load(os.path.join(self.checkpoint_dir, "epsilon_history.npy"), allow_pickle=True))
            with open(os.path.join(self.checkpoint_dir, "iteration.txt"), "r") as f:
                self.iteration = int(f.read())
            print(f"Checkpoint loaded from iteration {self.iteration}")
        except FileNotFoundError:
            print("No checkpoint found. Starting from scratch.")

    def upload_to_s3(self):
        """Uploads checkpoint files to an S3 bucket."""
        s3 = boto3.client('s3')
        for filename in os.listdir(self.checkpoint_dir):
            local_path = os.path.join(self.checkpoint_dir, filename)
            s3_key = f"jotto/{filename}"
            s3.upload_file(local_path, S3_BUCKET_NAME, s3_key)
            print(f"Uploaded {filename} to S3 bucket {S3_BUCKET_NAME} as {s3_key}")

    def download_from_s3(self):
        """Downloads checkpoint files from an S3 bucket."""
        if not S3_BUCKET_NAME:
            raise ValueError("S3_BUCKET_NAME is not configured.")

        s3 = boto3.client('s3')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        for obj in s3.list_objects(Bucket=S3_BUCKET_NAME, Prefix="jotto/")['Contents']:
            s3_key = obj['Key']
            local_path = os.path.join(self.checkpoint_dir, os.path.basename(s3_key))
            s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
            print(f"Downloaded {s3_key} to {local_path}")

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