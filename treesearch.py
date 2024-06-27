import numpy as np
import tqdm
import copy

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
    def is_terminal(self):
        pass
    @abstractmethod
    def get_outcome(self):
        pass
    @abstractmethod
    def get_actionspace(self):
        pass


class MCTS():
    class Node():
        def __init__(self, parent, action):
            self.parent = parent
            self.children = []

            self.action = action
            self.visits = 0

            self.value_sum = 0.
        
        def calcUcb(self, exploration_bias):
            n = self.visits
            N = self.parent.visits

            v = self.value_sum / self.visits
            u = exploration_bias * np.sqrt(np.log(N) / n)

            return v + u
        
        def select_child(self, exploration_bias):
            highest_ucb = -np.inf

            for c in self.children:
                c_ucb = c.calcUcb(exploration_bias)

                if c_ucb > highest_ucb:
                    highest_ucb = c_ucb
                    child = c
            
            return child
        
        def is_fully_expanded(self, game_state):
            return len(self.children) < len(game_state.possible_moves)
         
        def traverse(self, game_state, exploration_bias):
            current_node = self

            while current_node.is_fully_expanded() and not game_state.is_terminal():
                current_node = current_node.select_child(exploration_bias)

                game_state.push(current_node.action)
            
            return current_node
        
        def expand(self, game_state):
            actionspace = game_state.get_actionspace()
            action = actionspace[len(self.children)]

            self.children.append(MCTS.Node(self, action))

        def backprop(self, value, game_state):
            game_state.pop()

            self.value_sum += value
            self.parent.backprop(-value)
        
        def best_action(self):
            max_visits = -1

            for c in self.children:
                if c.visits > max_visits:
                    max_visits = c.visits
                    best_action = c.action
            
            return best_action
    
    def __init__(self, initial_game_state, exploration_bias = np.sqrt(2)):
        self.root = MCTS.Node(None, None)
        self.exploration_bias = exploration_bias
        self.game_state = initial_game_state
    
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
            if not self.game_state.is_terminal():
                selected_node = selected_node.expand(self.game_state)

            # Simulation
            value = self.value_fn(self.game_state)
            
            # Backpropagation
            selected_node.backprop(value)
        
        return self.root.best_action() 
    
    def observe_action(self, action):
        self.game_state.push(action)
        
        for c in self.root.children:
            if c.action == action:
                self.root = c
                return
            else:
                continue