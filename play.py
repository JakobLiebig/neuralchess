import torch
import os
import chess

import env
import treesearch
import value_fn
import value_net

nn_path = f'{os.getcwd()}/nns/v1.0.0.nn'
nn = value_net.Net()
nn.load_state_dict(torch.load(nn_path))

val_fn = value_fn.Func(nn, weight=1.8)

tree = treesearch.TreeSearch(val_fn)

game = env.ChessEnv()

search_depth = 3

while not game.is_over():
    if game.board.turn:
        print(game.board)
        
        while True:
            move_str = input('input move (UCI)>')

            try:
                move = chess.Move.from_uci(move_str)

                if move not in game.board.legal_moves:
                    print('Please enter a legal move!')
                else:
                    break
            except chess.InvalidMoveError:
                print('Please enter a valid uci move!')
            
        game.push(move)
    else:
        _, move = tree.alphabeta(game, search_depth, False)
        
        print(f'Neuralchess plays {move.uci()}')
        game.push(move)