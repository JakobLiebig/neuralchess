import torch
import os
import chess

import env
import treesearch
import value_fn
import value_net

def get_move_dialog(legal_moves):
    while True:
        move_str = input('>')

        try:
            move = chess.Move.from_uci(move_str)

            if move not in legal_moves:
                print('Please enter a legal move!')
            else:
                return move
        except chess.InvalidMoveError:
            print('Please enter a valid UCI move!')
        
def main():
    nn_path = f'{os.getcwd()}/nns/v1.0.0.nn'
    nn = value_net.Net()
    nn.load_state_dict(torch.load(nn_path))

    val_fn = value_fn.Func(nn, weight=1.8)

    tree = treesearch.TreeSearch(val_fn)

    game = env.ChessEnv()

    search_depth = 3

    print('Welcome to neural-chess!')
    print('')
    print('Please Note: Enter moves using the UCI format')
    print('')

    while not game.is_over():
        if game.board.turn:
            print(game.board)
            
            move = get_move_dialog(game.board.legal_moves)
            game.push(move)
        else:
            _, move = tree.alphabeta(game, search_depth, False)
            game.push(move)

            print(f'Neuralchess plays {move.uci()}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Bye bye')