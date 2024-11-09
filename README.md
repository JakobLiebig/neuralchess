# Neural Chess
Neural chess is a neural-network-powered chess engine.

It utilizes Monte Carlo tree search and a value network to find optimal policies.
Model architecture, learning parameters, and board representation used are based on the methods suggested by [Matthia Sabatelli et al. 2018](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf).

*"The Bitmap Input represents all the 64 squares of
the board through the use of 12 binary features. Each
of these features represents one particular chess piece
and which side is moving it. A piece is marked with
0 when it is not present on that square, with 1 when it
belongs to the player who should move, and with −1
when it belongs to the opponent. The representation
is a binary sequence of bits of length 768 that is able
to represent the full chess position. There are in fact
12 different piece types and 64 total squares which
results in 768 inputs."*

*"We have used a three hidden layer deep MLP with
1048, 500 and 50 hidden units for layers 1, 2, and 3
respectively. In order to prevent overfitting a Dropout
regularization value of 20% on every layer has been
used. Each hidden layer is connected with a non-
linear activation function: the 3 main hidden layers
make use of the Rectified Linear Unit (ReLU) activa-
tion function, while the final output layer consists of
a Softmax output. The Adam algorithm has been used
for the stochastic optimization problem and has been
initialized with the following parameters: η = 0.001;
β1 = 0.90; β2 = 0.99 and ε = 1e − 0.8. The network
has been trained with Minibatches of 128 samples."*

### Color-based vs. turn-based board representation
The mentioned paper uses a turn-based approach to classify positions.
In this turn-based approach, the nn maps a chess position to a winning color.

My turn-based approach mirrors the board to always be the pov of the active player.
Then it maps that mirrored board to "winning for active player" or "losing for active player" (or "drawing").
My idea was, that a position where its whites turn favoring white would be favoring black if the piece colors would be swapped and it was blacks turn.
This also minimizes the number of possible unique inputs into the network. 

![alt text](https://github.com/JakobLiebig/neuralchess/blob/main/docs/plt.png)

Using my limited amount of data and hardware, the turn-based approach outperformed the color-based approach. 

### Data
The Model learns through the [Lichess EVALUATIONS](https://database.lichess.org/#evals) data set.
It consists of 13,123,859 positions evaluated by stockfish. Each position is labeled as suggested in the mentioned paper:

*"A label of Winning has been assigned if cp > 1.5, Losing if it
was < −1.5 and Draw if the cp evaluation was
between these 2 values"*

The total number of positions is reduced to compensate for the uneven distribution of outcomes. (white: 27%, black: 14%, draw: 59%)

### Performance
Neuralchess plays very poorly against humans. Its performance is best in the early stages of the game where it very quickly reaches a position where it's very confident it is winning.
After reaching such a position its performance drops drastically and it often loses.

```
[Event "Jakob vs Neuralchess Round 1"]
[Date "2024.06.30"]
[White "Jakob"]
[Black "Neuralchess"]
[Result "1-0"]

1.e4 f6 2.Nf3 Kf7 3.Bc4+ e6 4.d4 Bb4+ 5.c3 Ba5 6.d5 c5 7.dxe6+ Ke7 8.e5 dxe6 9.Qxd8+ Bxd8 10.exf6+ gxf6 11.b4 Ba5 12.bxa5 Nd7 13.Ba3 e5 14.Bb5 Ke6 15.c4 Rb8 16.Nc3 e4 17.Nxe4 a6 18.Bxd7+ Kxd7 19.Bxc5 f5 20.O-O-O+ Ke6 21.Neg5+ Kf6 22.Rd8 h6 23.Rd6+ Ke7 24.Re6+ Kd8 25.Bb6+ Kd7 26.Ne5#
```

I think there are two reasons for its bad play:
1. **Limited amount of data and hardware (test accuracy of ~80%)**
2. **Using a classifier as a value function:**
   Being a classifier, the value network was trained to give very confident approximations leading to little to no nuance in its evaluations.
   Once neuralchess has reached a position with a high probability of winning, it evaluates nearly all positions after the same.
   For example: Winning or losing a piece after a "winning" position are both evaluated the same.

### Usage Example for Linux
```
git clone https://github.com/JakobLiebig/neuralchess.git
cd neuralchess
python3 -m venv ./.venv && source /.venv/bin/activate   # setup virtual environment
pip install -r requirements.txt
python3 play.py

# play using play.ipynb
```
