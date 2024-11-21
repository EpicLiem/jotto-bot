# play_with_guesser.py

from jotto_game import JottoGame
from strategy_oracle import StrategyOracle
from utils import load_strategy
from constants import WORD_LENGTH, HIDER_STRATEGY_FILE, STRATEGY_HISTORY_FILE
import sys


def main():
    # Load the word length (ensure it matches the one used in training)
    word_length = WORD_LENGTH

    # Load the dictionary
    game = JottoGame(word_length)
    dictionary = game.dictionary
    word_to_index = {word: idx for idx, word in enumerate(dictionary)}

    # Prompt the user to enter a secret word
    secret_word = input(f"Enter a secret word of length {word_length}: ").strip().lower()
    if secret_word not in word_to_index:
        print("Invalid word. Please enter a valid word from the dictionary.")
        print(word_to_index)
        return
    secret_word_index = word_to_index[secret_word]

    # Load the hider's strategy
    try:
        hider_strategy = load_strategy(HIDER_STRATEGY_FILE)
    except FileNotFoundError:
        print("Hider strategy not found. Ensure that the strategy has been computed and saved.")
        return

    # Load the strategy history
    try:
        import numpy as np
        strategy_history = np.load(STRATEGY_HISTORY_FILE, allow_pickle=True)
    except FileNotFoundError:
        print("Strategy history not found. Ensure that the strategy history has been saved.")
        return

    # Initialize the oracle with the loaded strategies
    oracle = StrategyOracle(game)
    oracle.hider_strategy = hider_strategy
    oracle.iteration_history = list(strategy_history)

    # Start the guessing game
    game.reset_state()
    state = game.state.copy()
    num_guesses = 0

    print("\nStarting the guessing game...\n")
    while True:
        num_guesses += 1
        # Get the guess from the guesser's strategy
        guess_index = oracle.sample_guesser_strategy(state)
        guess_word = dictionary[guess_index]

        # Provide feedback
        feedback = game.num_common_letters(guess_index, secret_word_index)
        print(f"Guess {num_guesses}: {guess_word.upper()} - Feedback: {feedback}")

        if feedback == word_length:
            print(
                f"\nThe guesser has correctly guessed the secret word '{secret_word.upper()}' in {num_guesses} guesses!")
            break

        # Update the game state based on the feedback
        game.update_state(guess_index, feedback)
        state = game.state.copy()

        # Check if no possible words remain (should not happen if strategies are correct)
        if not state.any():
            print("No possible words remain. The guesser cannot guess the word.")
            break


if __name__ == "__main__":
    main()