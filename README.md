# Neural Chess

A neural-network-powered chess engine that combines deep learning with Monte Carlo tree search to find optimal move policies.

Built on [Sabatelli et al. 2018](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf), with a novel turn-based board representation that improves the performance of the value network.

## Quick Start

```bash
git clone https://github.com/JakobLiebig/neuralchess.git
cd neuralchess
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 play.py              # Play via CLI
jupyter notebook play.ipynb  # Interactive notebook
```

## Architecture

**Network design:**
- 3-layer MLP with 1048, 500, and 50 hidden units
- ReLU activation on all hidden layers
- 20% dropout per layer for regularization
- Softmax output over 3 classes (winning, drawing, losing)

**Training configuration:**
- Optimizer: Adam with η=0.0001, β₁=0.90, β₂=0.999, ε=1e-8
- Batch size: 128
- Loss function: categorical crossentropy

## Board Representation

### Input Encoding

Each board position is represented as a 768-dimensional binary vector:

- **64 squares** × **12 features** (6 piece types × 2 players: active player and opponent)
- Each feature is `1` if the corresponding piece stands on that square, `0` otherwise
- This compactly captures any legal chess position

### Turn-Based vs. Color-Based Representation

The paper uses a **color-based** approach: positions are always mapped to White's perspective, and the network answers "does White win, lose, or draw?"

This engine uses a **turn-based** approach instead: the board is mirrored so positions are always evaluated from the active player's perspective, and the network answers "does the active player win, lose, or draw?"

**Why this matters:**
- It exploits symmetry: a position favoring White on White's move mirrors a position favoring Black on Black's move
- The model only needs to learn one pattern instead of two, reducing the required input diversity

Given limited data and resources, training on the turn-based representation outperforms the color-based one.

![Board representation comparison](https://github.com/JakobLiebig/neuralchess/blob/main/docs/plt.png)

## Training Data

**Dataset:** [Lichess Evaluations](https://database.lichess.org/#evals), 13.1 million positions evaluated by Stockfish.

**Position labels:**
- **Winning:** evaluation > 1.5 (27% of dataset)
- **Losing:** evaluation < -1.5 (14% of dataset)
- **Drawing:** -1.5 ≤ evaluation ≤ 1.5 (59% of dataset)

**Class balancing:** The dataset is heavily skewed toward draws, so positions are resampled to create a balanced training distribution.

## Results

**Test accuracy:** ~80%

**Gameplay:** The engine plays poorly against humans. It is strongest in the opening, where it quickly reaches positions it evaluates as advantageous, but its play drops off significantly in the middlegame and endgame, often squandering early advantages.

**Sample game (Jakob vs. Neuralchess, 1-0):**
```
[Event "Jakob vs Neuralchess"]
[Date "2024.06.30"]
[White "Jakob"]
[Black "Neuralchess"]
[Result "1-0"]

1.e4 f6 2.Nf3 Kf7 3.Bc4+ e6 4.d4 Bb4+ 5.c3 Ba5 6.d5 c5 7.dxe6+ Ke7
8.e5 dxe6 9.Qxd8+ Bxd8 10.exf6+ gxf6 11.b4 Ba5 12.bxa5 Nd7 13.Ba3 e5
14.Bb5 Ke6 15.c4 Rb8 16.Nc3 e4 17.Nxe4 a6 18.Bxd7+ Kxd7 19.Bxc5 f5
20.O-O-O+ Ke6 21.Neg5+ Kf6 22.Rd8 h6 23.Rd6+ Ke7 24.Re6+ Kd8 25.Bb6+ Kd7 26.Ne5#
```

## Why Performance Lags

### 1. Limited Data and Hardware

At ~80% test accuracy there is substantial room for improvement. More compute and larger training datasets would directly improve evaluation accuracy.

### 2. Classifier vs. Regressor Design

The value network is trained as a **3-class classifier** rather than a continuous **regressor**:

- Classification produces confidence scores that saturate at high certainty
- Once a position reaches a high "winning" probability, nearly all subsequent positions receive almost identical evaluations
- Subtle differences between positions are lost: being up a pawn and being up a piece both output "winning"
- A regressor would output continuous value estimates (e.g. +2.5 vs. +6.0 pawns), enabling better play