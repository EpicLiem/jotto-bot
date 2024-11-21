# precompute.py

from utils import load_dictionary, precompute_common_letters
from constants import WORD_LENGTH

def main():
    dictionary = load_dictionary(WORD_LENGTH)
    precompute_common_letters(dictionary)
    print("Common letters matrix precomputed and saved.")

if __name__ == "__main__":
    main()