# Neural Chess

A neural-network-powered chess engine that combines deep learning with Monte Carlo tree search to find optimal move policies.

Built on [Sabatelli et al. 2018](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf). A novel turn-based board representation improves performance of the value network.

## Quick Start

```bash
git clone https://github.com/JakobLiebig/neuralchess.git
cd neuralchess
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 play.py              # Play via CLI
jupyter notebook play.ipynb  # Interactive notebook
```

## How It Works

### Architecture

**Network Design:**
- 3-layer deep MLP with 1048, 500, and 50 hidden units
- ReLU activation on all hidden layers
- 20% dropout per layer for regularization
- Softmax output for 3-class probability distribution (Winning/Drawing/Losing)

**Training Configuration:**
- Optimizer: Adam with η=0.001, β₁=0.90, β₂=0.99, ε=1e⁻⁸
- Batch size: 128 samples
- Loss function: Categorical crossentropy

## Board Representation

### Binary Input Encoding

The engine represents each board position as a 768-dimensional binary vector:

- **64 squares** × **12 features** (6 piece types × 2 colors)
- Each square encoded as: `1` (active player's piece), `-1` (opponent's piece), or `0` (empty)
- Compact yet comprehensive representation of any legal chess position

### Turn-Based vs Color-Based Representation

The paper uses a **color-based approach**: always maps positions to white's perspective (does white win/lose/draw?)

Our **turn-based approach**: mirrors the board so positions are always evaluated from the active player's perspective (does the active player win/lose/draw?)

**Why this matters:**
- Exploits symmetry: a position favoring White on White's move mirrors a position favoring Black on Black's move
- The model only needs to learn one pattern instead of two, reducing required input diversity

Given limited data and resources, training on turn based representations outperforms training on color based.
![Board representation comparison](https://github.com/JakobLiebig/neuralchess/blob/main/docs/plt.png)

## Training Data

**Dataset:** [Lichess Evaluations](https://database.lichess.org/#evals) - 13.1 million positions evaluated by Stockfish

**Position Labels:**
- **Winning:** centipawn evaluation > 1.5 (27% of dataset)
- **Losing:** centipawn evaluation < -1.5 (14% of dataset)
- **Draw:** -1.5 ≤ centipawn ≤ 1.5 (59% of dataset)

**Class Balancing:** Dataset exhibits extreme class imbalance toward draws. Positions are resampled to create balanced training distributions.

## Results

**Test Accuracy:** ~80%

**Gameplay Performance:** The engine plays poorly against humans. It shows strength in the opening phase where it quickly reaches positions it evaluates as advantageous. However, performance drops significantly in the middlegame and endgame, often squandering early advantages.

**Sample Game (White vs Neuralchess — White wins):**
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

### 1. Limited Data & Hardware

With only ~80% test accuracy, there's substantial room for improvement. More computational resources and larger training datasets would directly improve evaluation accuracy.

### 2. Classifier vs Regressor Design

The value network is trained as a **3-class categorical classifier** rather than a continuous **regressor**:

- Classification produces binary-like confidence scores that saturate at high certainty
- Once a position achieves high "winning" probability, nearly all subsequent positions receive nearly identical evaluations
- Cannot represent subtle differences between positions (e.g., up a pawn vs up a piece both output "winning")
- A regressor would output continuous value estimates (e.g., +2.5 vs +6.0 pawns) enabling better play