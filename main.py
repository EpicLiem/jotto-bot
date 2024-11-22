# main.py

from jotto_game import JottoGame
from strategy_oracle import StrategyOracle
from fictitious_play import FictitiousPlay
from constants import WORD_LENGTH, NUM_ITERATIONS
import sys

def main():
    if len(sys.argv) > 1:
        word_length = int(sys.argv[1])
    else:
        word_length = WORD_LENGTH
    # Initialize game and oracle
    game = JottoGame(word_length)
    oracle = StrategyOracle(game)
    # Run fictitious play
    fp = FictitiousPlay(game, oracle, NUM_ITERATIONS)
    fp.download_from_s3()
    fp.run()
    # Sample a game
    game.reset_state()
    state = game.state.copy()
    guess_index = oracle.sample_guesser_strategy(state)
    guess_word = game.dictionary[guess_index]
    print(f"Guesser's first guess: {guess_word}")

if __name__ == "__main__":
    main()