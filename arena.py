import chess
import chess.pgn

from abc import abstractmethod, ABC
from typing import Tuple

class PlayerBase(ABC):
    @abstractmethod
    def get_move(self, game_state):
        pass
    @abstractmethod
    def observe(self, action):
        pass

class Arena():
    def pit(self, players: tuple[PlayerBase, PlayerBase], p1_play_as, init_pos=None):
        p1, p2 = players
        game_state = chess.Board() if init_pos == None else init_pos

        while not game_state.is_game_over():
            if game_state.turn == p1_play_as:
                action = p1.get_move(game_state)
            else:
                action = p2.get_move(game_state)
            
            p1.observe(action)
            p2.observe(action)

            game_state.push(action)

        return game_state.outcome()