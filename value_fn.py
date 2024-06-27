import torch
import env

class NNValueFn():
    def __init__(self, nn, weight:float):
        self.nn = nn
        self.weight = weight
    
    def __call__(self, game: env.ChessEnv):
        piece_map = game.board.piece_map()
        is_white_turn = game.board.turn

        x = value_net.Encoder.from_piece_map(piece_map, is_white_turn).view((1, 2*6*8*8))
        with torch.no_grad():
            y = self.nn(x).flatten().numpy()

        white_win_pct = y[0]
        draw_pct = y[2]
        
        # weighted sum
        value = white_win_pct * self.weight + draw_pct
        return value