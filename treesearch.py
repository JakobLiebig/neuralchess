import numpy as np
from abc import ABC, abstractmethod

class GameBase(ABC):
    @abstractmethod
    def legal_moves(self):
        pass
    @abstractmethod
    def push(self, move):
        pass
    @abstractmethod
    def pop(self):
        pass
    @abstractmethod
    def is_over(self):
        pass
    @abstractmethod
    def get_outcome(self):
        pass


class TreeSearch():
    def __init__(self, value_fn):
        self.value_fn = value_fn

    def minimax(self, game: GameBase, depth: int, is_max: bool):
        if depth == 0:
            return self.value_fn(game), None
        elif game.is_over():
            return game.get_outcome(), None

        best_val = None
        best_move = None

        for move in game.legal_moves():
            game.push(move)

            val, _ = self.minimax(game, depth-1, not is_max)

            if (best_val == None) or (is_max and val > best_val) or (not is_max and val < best_val):
                best_val = val
                best_move = move
            
            game.pop()
        
        return best_val, best_move
    

    def alphabeta(self, game: GameBase, depth: int, is_max: bool, alpha=-np.inf, beta=np.inf):
        if depth == 0:
            return self.value_fn(game), None
        elif game.is_over():
            return game.get_outcome(), None

        best_val = None
        best_move = None

        if is_max:
            for move in game.legal_moves():
                game.push(move)

                val, _ = self.alphabeta(game, depth-1, False, alpha, beta)
                
                if best_val == None or val > best_val:
                    best_val = val
                    best_move = move
                    alpha = max(alpha, best_val)
                
                game.pop()

                if beta <= alpha:
                    break
        else:
            for move in game.legal_moves():
                game.push(move)

                val, _ = self.alphabeta(game, depth-1, True, alpha, beta)
                
                if best_val == None or val < best_val:
                    best_val = val
                    best_move = move
                    beta = min(beta, best_val)
                
                game.pop()

                if beta <= alpha:
                    break
        
        return best_val, best_move