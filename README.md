# Neural chess
My goal with this project is to build a neural network-powered chess bot.

The chess bot will utilize minimax with alpha-beta pruning and a value network to find optimal policies.
The structure of the model, learning parameters, and board representation used for the value network are based on the methods suggested by [Matthia Sabatelli et al. 2018](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf).

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

### Dataset
The Model will learn through the [Lichess EVALUATIONS](https://database.lichess.org/#evals) data set.
It consists of 13,123,859 positions evaluated by stockfish. Those evaluations are turned into labels as suggested in the mentioned paper:

*"A label of Winning has been assigned if cp > 1.5, Losing if it
was < −1.5 and Draw if the cp evaluation was
between these 2 values"*

The total number of positions was reduced to compensate for the uneven distribution of outcomes in the data set. (white: 27%, black: 14%, draw: 59%)
