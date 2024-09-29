
import torch
import numpy as np
import tqdm
import copy
import chess

import arena
import nn.Encoder
from nn.Classifier import ChessClassifier

class MCTS():
    class Node():
        def __init__(self, parent, action, legal_moves):
            self.parent = parent
            self.children = []
            self.unexplored_actions = list(legal_moves)

            self.action = action
            self.visits = 0

            self.value_sum = 0.
        
        def calcUcb(self, exploration_bias):
            n = self.visits
            N = self.parent.visits

            v = -self.value_sum / self.visits # - self.value_sum because the ucb is calculated for the opponents view
            u = exploration_bias * np.sqrt(np.log(N) / n)

            return v + u
        
        def select_child(self, exploration_bias):
            highest_ucb = None

            for c in self.children:
                c_ucb = c.calcUcb(exploration_bias)

                if highest_ucb == None or c_ucb > highest_ucb:
                    highest_ucb = c_ucb
                    child = c
            
            return child
        
        def is_fully_expanded(self):
            return len(self.unexplored_actions) <= 0
         
        def traverse(self, game_state, exploration_bias):
            current_node = self
            self.visits += 1

            while current_node.is_fully_expanded() and not game_state.is_game_over():
                current_node = current_node.select_child(exploration_bias)
                
                current_node.visits += 1
                game_state.push(current_node.action)
            
            return current_node
        
        def expand(self, game_state):
            action = self.unexplored_actions.pop()
            game_state.push(action)

            new_node = MCTS.Node(self, action, game_state.legal_moves)
            self.children.append(new_node)

            new_node.visits += 1

            return new_node

        def backprop(self, value, game_state):
            self.value_sum += value
            
            if self.parent != None:
                game_state.pop()
                self.parent.backprop(-value, game_state)
        
        def best_action(self):
            max_visits = -1

            for c in self.children:
                if c.visits > max_visits:
                    max_visits = c.visits
                    best_action = c.action
            
            return best_action
    
    def __init__(self, value_fn, init_pos, exploration_bias = np.sqrt(2)):
        self.root = MCTS.Node(None, None, init_pos.legal_moves)
        
        self.value_fn = value_fn
        self.game_state = init_pos
        self.exploration_bias = exploration_bias
        
    def search(self, depth, verbose=True):
        if verbose:
            iterator = tqdm.trange(depth)
        else:
            iterator = range(depth)

        for i in iterator:            
            # Selection
            # while traversing the tree each node pushes its action to the game_state to minimize the need for saving each gamestate
            selected_node = self.root.traverse(self.game_state, self.exploration_bias)
            
            # Expension
            if not self.game_state.is_game_over():
                selected_node = selected_node.expand(self.game_state)

                # Simulation
                value = self.value_fn(self.game_state)
            else:
                # game is over take actual results
                outcome = self.game_state.outcome()
                is_draw = outcome.winner == None
                value = -np.inf if not is_draw else 0. 

            # Backpropagation
            selected_node.backprop(value, self.game_state)
        
        return self.root.best_action() 
    
    def observe_action(self, action):
        self.game_state.push(action)
        
        for child in self.root.children:
            if child.action == action:
                self.root = child
                self.root.parent = None
                return
            else:
                continue
        
        # action is not yet part of the gametree!
        self.root = MCTS.Node(None, action, self.game_state.legal_moves)

class ValueFunc():
    def __init__(self, path_to_nn:str, weight:float):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(path_to_nn, map_location=device)['state_dict']
        
        self.nn = ChessClassifier(in_features=2*6*8*8, labels=3)
        self.nn.load_state_dict(state_dict)

        self.weight = weight
    
    def __call__(self, game_state):
        fen = game_state.fen()
        features = nn.Encoder.TurnBased.fen_to_tensor(fen)
        logits = self.nn.predict(features).numpy()
        
        win_pct = logits[nn.Encoder.TurnBased.label_dict['Win']]
        draw_pct = logits[nn.Encoder.TurnBased.label_dict['Draw']]

        # weighted sum
        return win_pct + self.weight * draw_pct
        

    def get_eval_dict(self, game_state):
        fen = game_state.fen()
        features = nn.Encoder.TurnBased.fen_to_tensor(fen)
        logits = self.nn.predict(features).numpy()
        
        eval_dict = {}

        for k in nn.Encoder.TurnBased.label_dict:
            eval_dict[k] = logits[nn.Encoder.TurnBased.label_dict[k]]
        
        return eval_dict

class NeuralChess(arena.PlayerBase):
    def __init__(self,
                 path_to_nn:str,
                 search_iterations:int,
                 weight=0.1,
                 exploration_bias=np.sqrt(2),
                 init_pos=None,
                 verbose=True):
        self.search_iter = search_iterations
        self.verbose = verbose
        self.val_fn = ValueFunc(path_to_nn, weight)

        init_pos = chess.Board() if init_pos == None else init_pos
        
        self.tree = MCTS(self.val_fn, init_pos, exploration_bias)
    

    def get_move(self, game_state):
        fen = game_state.fen()
        eval_dict = self.val_fn.get_eval_dict(game_state)
    
        if self.verbose:
            print(f'Neural chess evaluates its position as:')
            for key in eval_dict:
                print(f'{key}: {round(eval_dict[key] * 100., 5)}')
        
        move = self.tree.search(self.search_iter, verbose=self.verbose)
        return move
    
    def observe(self, action):
        self.tree.observe_action(action)