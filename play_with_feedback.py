import numpy as np
from jotto_game import JottoGame
from strategy_oracle import StrategyOracle
from constants import WORD_LENGTH, HIDER_STRATEGY_FILE, STRATEGY_HISTORY_FILE
from utils import load_strategy

def main():
    print(f"Welcome to the Jotto game with a guessing bot!")
    print(f"The bot will try to guess your secret word of length {WORD_LENGTH}.")
    print("Provide feedback after each guess (number of common letters).\n")

    # Load the game and dictionary
    game = JottoGame(WORD_LENGTH)
    dictionary = game.dictionary
    word_to_index = {word: idx for idx, word in enumerate(dictionary)}

    # Load the full dictionary with anagrams
    try:
        with open('data/dictionary.txt', 'r') as f:
            full_dictionary = [line.strip().lower() for line in f if len(line.strip()) == WORD_LENGTH]
    except FileNotFoundError:
        print("Full dictionary not found. Ensure that 'data/dictionary.txt' exists.")
        return

    # Build an anagram mapping from sorted letters to words
    anagram_dict = {}
    for word in full_dictionary:
        key = ''.join(sorted(word))
        anagram_dict.setdefault(key, []).append(word)

    # Load precomputed strategies
    try:
        hider_strategy = load_strategy(HIDER_STRATEGY_FILE)
        strategy_history = np.load(STRATEGY_HISTORY_FILE, allow_pickle=True)
    except FileNotFoundError:
        print("Precomputed strategies not found. Please ensure the strategies are generated first.")
        return

    # Initialize the strategy oracle
    oracle = StrategyOracle(game)
    oracle.hider_strategy = hider_strategy
    oracle.iteration_history = list(strategy_history)

    # Start the game
    state = game.state.copy()
    num_guesses = 0
    guessed_words = set()
    print("Think of a secret word and keep it in your mind.")
    print(f"When the bot guesses, enter the feedback (0-{WORD_LENGTH} for {WORD_LENGTH}-letter words, or the number of common letters).")
    print("If the bot's guess matches all letters, indicate whether it's the correct word (yes/no).\n")

    while True:
        # Make a guess using the oracle
        num_guesses += 1
        guess_index = oracle.sample_guesser_strategy(state)
        guess_word = dictionary[guess_index]
        guessed_words.add(guess_word)
        print(f"Guess {num_guesses}: {guess_word.upper()}")

        # Get feedback from the user
        try:
            feedback = int(input(f"Enter feedback (0-{WORD_LENGTH}): "))
            if feedback < 0 or feedback > WORD_LENGTH:
                raise ValueError
        except ValueError:
            print(f"Invalid feedback. Please enter a number between 0 and {WORD_LENGTH}.")
            continue

        # Update the game state with the feedback
        game.update_state(guess_index, feedback)
        state = game.state.copy()

        # Check if no valid guesses remain
        if not state.any():
            print("\nNo valid guesses remain. Did you make an error in providing feedback?")
            break

        # Handle maximum feedback
        if feedback == WORD_LENGTH:
            # Ask if the guessed word is the secret word
            is_correct = input(f"Is '{guess_word.upper()}' your secret word? (yes/no): ").strip().lower()
            if is_correct == 'yes':
                print(f"\nThe bot has correctly guessed your word '{guess_word.upper()}' in {num_guesses} guesses!")
                break
            else:
                # Find all anagrams of the guessed word
                key = ''.join(sorted(guess_word))
                anagrams = anagram_dict.get(key, [])
                # Remove already guessed words
                remaining_anagrams = [word for word in anagrams if word not in guessed_words]
                if not remaining_anagrams:
                    print("\nNo more anagrams to guess. The bot couldn't find your word.")
                    break
                print("\nAll letters match but the word is not correct. Continuing to guess anagrams...\n")
                # Guess remaining anagrams
                for anagram in remaining_anagrams:
                    num_guesses += 1
                    guessed_words.add(anagram)
                    print(f"Guess {num_guesses}: {anagram.upper()}")
                    # Ask if the guessed anagram is the secret word
                    is_correct = input(f"Is '{anagram.upper()}' your secret word? (yes/no): ").strip().lower()
                    if is_correct == 'yes':
                        print(f"\nThe bot has correctly guessed your word '{anagram.upper()}' in {num_guesses} guesses!")
                        return
                print("\nThe bot couldn't find your word among the anagrams.")
                break

if __name__ == "__main__":
    main()