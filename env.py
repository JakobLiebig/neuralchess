import chess
import numpy as np

from treesearch import GameBase

class ChessEnv(GameBase):
    def __init__(self):
        self.board = chess.Board()
    
    def push(self, move):
        self.board.push(move)
    
    def pop(self):
        self.board.pop()
    
    def legal_moves(self):
        return self.board.legal_moves
    
    def is_over(self):
        return self.board.outcome() != None
    
    def get_outcome(self):
        outcome = self.board.outcome()
        
        if outcome == None:
            return None

        if outcome.winner == chess.WHITE:
            return +np.inf
        elif outcome.winner == chess.BLACK:
            return -np.inf
        else:
            # Draw
            return 0