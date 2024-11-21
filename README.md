# Jotto Game Solver

This project implements an algorithm to compute strong game-theoretic strategies in **Jotto**, a word-guessing game. The implementation uses two asymmetrical reinforcement learning agents: a **Guesser** and a **Secret Keeper (Hider)**. The Guesser aims to deduce the secret word with minimal guesses, while the Hider selects a word that maximizes the Guesser's difficulty.

The algorithm is based on the paper:

> **Ganzfried, Sam. "Computing Strong Game-Theoretic Strategies in Jotto."** *Advances in Computer Games*, Springer, 2011, pp. 282–294.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Precomputing Data](#precomputing-data)
  - [Training Strategies](#training-strategies)
  - [Interactive Guessing Game](#interactive-guessing-game)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Introduction

**Jotto** is a two-player word game where one player (the Guesser) tries to deduce the secret word chosen by the other player (the Hider) by making guesses and receiving feedback on the number of common letters.

Due to the immense state space of Jotto (approximately $begin:math:text$10^{853}$end:math:text$ states for five-letter words), traditional game-solving methods are infeasible. This project adopts the **oracular strategy representation** and extends the **fictitious play algorithm** to compute approximate equilibrium strategies efficiently.

---

## Features

- **Asymmetrical Agents**: Implements two agents with different roles and strategies.
- **Oracular Strategy Representation**: Efficiently represents strategies without explicitly storing the entire strategy space.
- **Fictitious Play Algorithm**: Computes approximate equilibrium strategies through iterative best responses.
- **Parallel Computing**: Utilizes multiprocessing to handle large computations.
- **Interactive Gameplay**: Allows users to play against the trained Guesser agent.
- **Configurable Word Lengths**: Supports word lengths from 2 to 5 letters.

---

## Project Structure

```
jotto/
├── data/
│   ├── dictionary.txt        # Preprocessed dictionary file
│   ├── common_letters.npy    # Precomputed common letters matrix
├── strategies/
│   ├── hider_strategy.npy    # Hider's strategy vector
│   ├── strategy_history.npy  # Hider's strategy over iterations
├── main.py                   # Main script for training strategies
├── play_with_guesser.py      # Interactive script to play against the Guesser
├── precompute.py             # Script to precompute necessary data
├── jotto_game.py             # Game logic and mechanics
├── strategy_oracle.py        # Oracular strategies implementation
├── fictitious_play.py        # Fictitious play algorithm implementation
├── utils.py                  # Utility functions and helpers
├── constants.py              # Constants and configurations
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
```

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/jotto.git
   cd jotto
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dictionary**

   - Ensure you have a `dictionary.txt` file in the `data/` directory.
   - The file should contain a list of valid words, one per line.

---

## Usage

### Precomputing Data

Before running the main training script, precompute the common letters matrix:

```bash
python precompute.py
```

### Training Strategies

Run the main script to compute and save the strategies:

```bash
python main.py
```

- **Note**: The default word length is set in `constants.py`. To use a different word length, modify the `WORD_LENGTH` variable or pass it as a command-line argument:

  ```bash
  python main.py 5  # For 5-letter words
  ```

### Interactive Guessing Game

Play against the trained Guesser agent by running:

```bash
python play_with_guesser.py
```

- Enter a secret word when prompted.
- The Guesser will attempt to guess your word, showing each guess and feedback.

---

## Configuration

All configurations are set in the `constants.py` file:

- `WORD_LENGTH`: Length of words to use (default is 5).
- `NUM_ITERATIONS`: Number of iterations for the fictitious play algorithm.
- `DICTIONARY_FILE`: Path to the dictionary file.
- `HIDER_STRATEGY_FILE`: Path to save the hider's strategy.
- `STRATEGY_HISTORY_FILE`: Path to save the strategy history.

---

## Dependencies

The project requires the following Python packages:

- `numpy`
- `tqdm`

Install them using:

```bash
pip install -r requirements.txt
```

---

## Example

**Training Strategies**

```bash
python precompute.py
python main.py
```

**Sample Output:**

```
Common letters matrix precomputed and saved.
Starting training with word length 5...
100%|████████████████████████████| 5000/5000 [01:23<00:00, 60.01it/s]
Best Epsilon: 0.3342 at iteration 3456
```

**Playing the Game**

```bash
python play_with_guesser.py
```

**Sample Interaction:**

```
Enter a secret word of length 5: crane

Starting the guessing game...

Guess 1: AUDIO - Feedback: 1
Guess 2: STERN - Feedback: 3
Guess 3: SNARE - Feedback: 4
Guess 4: CRANE - Feedback: 5

The guesser has correctly guessed the secret word 'CRANE' in 4 guesses!
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with descriptive messages.
4. Open a pull request detailing your changes.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## References

This implementation is based on the following paper:

**Ganzfried, Sam. "Computing Strong Game-Theoretic Strategies in Jotto."** *Advances in Computer Games*, Springer, 2011, pp. 282–294.

BibTeX Citation:

```bibtex
@incollection{ganzfried2011computing,
  title={Computing strong game-theoretic strategies in Jotto},
  author={Ganzfried, Sam},
  booktitle={Advances in Computer Games},
  pages={282--294},
  year={2011},
  publisher={Springer}
}
```

---

**Note**: This project was for a cs project  and aims to replicate and demonstrate the methods described in the referenced paper. This repo is unlikely to be maintained.

---
