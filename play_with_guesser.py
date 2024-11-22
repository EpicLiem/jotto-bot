# play_with_guesser.py

from jotto_game import JottoGame
from strategy_oracle import StrategyOracle
from utils import load_strategy
from constants import WORD_LENGTH, HIDER_STRATEGY_FILE, STRATEGY_HISTORY_FILE
import sys
import numpy as np


def main():
    # Load the word length (ensure it matches the one used in training)
    word_length = WORD_LENGTH

    # Load the dictionary
    game = JottoGame(word_length)
    dictionary = game.dictionary
    word_to_index = {word: idx for idx, word in enumerate(dictionary)}

    # Load the full dictionary with anagrams
    try:
        with open('data/dictionary.txt', 'r') as f:
            full_dictionary = [line.strip().lower() for line in f if len(line.strip()) == word_length]
    except FileNotFoundError:
        print("Full dictionary not found. Ensure that 'data/dictionary.txt' exists.")
        return

    full_word_to_index = {word: idx for idx, word in enumerate(full_dictionary)}

    # Build an anagram mapping from sorted letters to words
    anagram_dict = {}
    for word in full_dictionary:
        key = ''.join(sorted(word))
        anagram_dict.setdefault(key, []).append(word)

    # Prompt the user to enter a secret word
    secret_word = input(f"Enter a secret word of length {word_length}: ").strip().lower()
    if secret_word not in full_word_to_index:
        print("Invalid word. Please enter a valid word from the full dictionary.")
        return

    # Map the secret word to the game's dictionary (without anagrams)
    if secret_word in word_to_index:
        secret_word_index = word_to_index[secret_word]
    else:
        # Find a word in the game's dictionary that is an anagram of the secret word
        secret_word_key = ''.join(sorted(secret_word))
        possible_indices = [word_to_index[word] for word in dictionary if ''.join(sorted(word)) == secret_word_key]
        if not possible_indices:
            print("The secret word cannot be mapped to the game's dictionary.")
            return
        secret_word_index = possible_indices[0]  # Choose the first matching anagram

    # Load the hider's strategy
    try:
        hider_strategy = load_strategy(HIDER_STRATEGY_FILE)
    except FileNotFoundError:
        print("Hider strategy not found. Ensure that the strategy has been computed and saved.")
        return

    # Load the strategy history
    try:
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
    guessed_words = set()

    print("\nStarting the guessing game...\n")
    while True:
        num_guesses += 1
        # Get the guess from the guesser's strategy
        guess_index = oracle.sample_guesser_strategy(state)
        guess_word = dictionary[guess_index]
        guessed_words.add(guess_word)

        # Provide feedback
        feedback = game.num_common_letters(guess_index, secret_word_index)
        print(f"Guess {num_guesses}: {guess_word.upper()} - Feedback: {feedback}")

        if feedback == word_length:
            if guess_word == secret_word:
                print(
                    f"\nThe guesser has correctly guessed the secret word '{secret_word.upper()}' in {num_guesses} guesses!")
                break
            else:
                # Find all anagrams of the guess word
                key = ''.join(sorted(guess_word))
                anagrams = anagram_dict.get(key, [])
                # Remove already guessed words
                remaining_anagrams = [word for word in anagrams if word not in guessed_words]
                if secret_word in remaining_anagrams:
                    print(f"\nAll letters match but the word is not correct. Continuing to guess anagrams...")
                    for anagram in remaining_anagrams:
                        num_guesses += 1
                        guessed_words.add(anagram)
                        print(f"Guess {num_guesses}: {anagram.upper()} - Feedback: {feedback}")
                        if anagram == secret_word:
                            print(
                                f"\nThe guesser has correctly guessed the secret word '{secret_word.upper()}' in {num_guesses} guesses!")
                            break
                    else:
                        # If the loop completes without finding the secret word
                        print("Error: Secret word not found among anagrams.")
                        break
                    break  # Break the outer loop after finding the secret word
                else:
                    # This should not happen
                    print("Error: Secret word not among anagrams of the guess word.")
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