import torch
import torch.nn as nn
import numpy as np

class Encoder():
    piece_dict = {
        'K' : 0,
        'Q' : 1,
        'R' : 2,
        'B' : 3,
        'N' : 4,
        'P' : 5
    }

    def from_fen(fen: str):
        pieces = torch.zeros([2, 6, 64])
        
        fen_sections = fen.split(' ')
        board_fen = fen_sections[0].replace('/', '')
        active_color = fen_sections[1]

        pos = 0
        for c in board_fen:
            if c.isdigit():
                pos += int(c)
            else:
                is_white = c.isupper()
                piece_str = c.upper()
                color_index = 0 if is_white else 1

                pieces[color_index, Encoder.piece_dict[piece_str], pos] = 1.
                pos += 1
        
        if active_color == 'w':
            pieces[1] *= -1
        else:
            pieces[0] *= -1
        
        pieces = pieces.flatten()
        
        return pieces

    def from_piece_map(piece_map, is_white_turn):
        pieces = torch.zeros([2, 6, 8*8])

        for position in piece_map:
            piece= piece_map[position]

            piece_str = str(piece).upper()
            color = 0 if piece.color else 1

            pieces[color, Encoder.piece_dict[piece_str], position] = 1.
        
        if is_white_turn:
            pieces[1] *= -1.
        else:
            pieces[0] *= -1.
        
        pieces = pieces.flatten()

        return pieces

class Sampler():
    def __init__(self, path_to_white, path_to_black, path_to_draw):
        with open(path_to_white, 'r') as file:
            self.white_db = file.read().split('\n')
        
        with open(path_to_black, 'r') as file:
            self.black_db = file.read().split('\n')
        
        with open(path_to_draw, 'r') as file:
            self.draw_db = file.read().split('\n')

    def gen_tensor(fen_batch):
        fen_tensor_list = []
        for fen in fen_batch:
            fen_tensor_list.append(Encoder.from_fen(fen))
        
        return torch.stack(fen_tensor_list)

    def gen_label(one_hot, size):
        label = torch.zeros([1, 3])
        label[0, one_hot] = 1.
        label = label.repeat(size, 1)
        
        return label

    def gen_uniform_sample(self, size):
        # generates sample containing equal quantities of outcomes
        # (while still matching size)
        white_size = size // 3
        black_size = size // 3
        draw_size = size -white_size-black_size

        white_fen_batch = np.random.choice(self.white_db, size=white_size, replace=False)
        white_x = Sampler.gen_tensor(white_fen_batch)
        white_label = Sampler.gen_label(0, white_size)

        black_fen_batch = np.random.choice(self.black_db, size=black_size, replace=False)
        black_x = Sampler.gen_tensor(black_fen_batch)
        black_label = Sampler.gen_label(1, black_size)

        draw_fen_batch = np.random.choice(self.draw_db, size=draw_size, replace=False)
        draw_x = Sampler.gen_tensor(draw_fen_batch)
        draw_label = Sampler.gen_label(2, draw_size)

        x = torch.cat([white_x, black_x, draw_x], 0)
        labels = torch.cat([white_label, black_label, draw_label], 0)

        return x, labels

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.func = nn.Sequential(
            nn.Linear(768, 1048),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1048, 500),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(500, 50),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(50, 3),
            nn.Softmax(1)
        )
        
    def __call__(self, x):
        return self.func(x)