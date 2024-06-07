import torch
import torch.nn as nn

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