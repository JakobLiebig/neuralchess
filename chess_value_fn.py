import torch
import env

import nn.Encoder
from nn.Classifier import ChessClassifier

class Func():
    def __init__(self, path_to_nn:str, weight:float):
        self.nn = ChessClassifier(in_features=2*6*8*8, labels=3)
        state_dict = torch.load(path_to_nn)['state_dict']
        self.nn.load_state_dict(state_dict)

        self.weight = weight
    
    def __call__(self, game: env.ChessEnv):
        fen = game.board.fen()

        features = nn.Encoder.TurnBased.fen_to_tensor(fen)
        logits = self.nn.predict(features)
        
        win_pct = logits[nn.Encoder.TurnBased.label_dict['Win']]
        draw_pct = logits[nn.Encoder.TurnBased.label_dict['Draw']]

        # weighted sum
        value = win_pct + self.weight * draw_pct
        return value