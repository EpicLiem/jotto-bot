# constants.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration variables
WORD_LENGTH = int(os.getenv("WORD_LENGTH", 5))
NUM_ITERATIONS = int(os.getenv("NUM_ITERATIONS", 5000))

DICTIONARY_FILE = os.getenv("DICTIONARY_FILE", "data/dictionary.txt")
COMMON_LETTERS_FILE = os.getenv("COMMON_LETTERS_FILE", "data/common_letters.npy")

HIDER_STRATEGY_FILE = os.getenv("HIDER_STRATEGY_FILE", "strategies/hider_strategy.npy")
STRATEGY_HISTORY_FILE = os.getenv("STRATEGY_HISTORY_FILE", "strategies/strategy_history.npy")

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "jottobot")