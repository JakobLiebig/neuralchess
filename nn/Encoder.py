import torch

class TurnBased:

    piece_dict = {'K':0, 'Q':1, 'R':2, 'B':3, 'N':4, 'P':5}
    label_dict = {'Win' : 0, 'Loss' : 1, 'Draw' : 2}

    def fen_to_tensor(fen):
        # Initialize a tensor of zeros with shape (2, 6, 8, 8)
        features = torch.zeros((2, 6, 8*8))
        
        # Split the FEN string into sections
        fen_sections = fen.split(' ')
        board_fen = fen_sections[0]
        active_color = fen_sections[1]
        mirror = active_color == 'b'

        pos_idx = 0
        for c in board_fen:
            if c.isdigit():
                pos_idx += int(c)
                continue
            elif c == '/':
                continue
            
            piece_color = c.isupper()
            piece_type = c.upper()
            
            piece_encoding = TurnBased.piece_dict[piece_type]
            
            # Determine the board position
            if mirror:
                activity_encoding = not piece_color
                position_encoding = 63 - pos_idx
            else:
                activity_encoding = piece_color
                position_encoding = pos_idx
            
            features[int(activity_encoding), piece_encoding, position_encoding] = 1.
            
            pos_idx+=1
        
        return features.view((2*6*8*8))

    def label_to_tensor(label):
        label_t = torch.zeros((3))

        label_t[TurnBased.label_dict[label]] = 1.

        return label_t

class ColorBased:

    piece_dict = {'K':0, 'Q':1, 'R':2, 'B':3, 'N':4, 'P':5}
    label_dict = {'White' : 0, 'Draw' : 1, 'Black' : 2}

    def fen_to_tensor(fen):
        features = torch.zeros((2, 6, 8 * 8))

        fen_sections = fen.split(' ')
        board_fen = fen_sections[0].replace('/', '')
        active_color = fen_sections[1]

        pos_index = 0
        for c in board_fen:
            if c.isdigit():
                pos_index += int(c)

            else:
                piece_color = c.isupper()
                piece_str = c.upper()

                activity_encoding = 1. if piece_color == active_color else -1.
                piece_encoding = ColorBased.piece_dict[piece_str]

                features[int(piece_color), piece_encoding, pos_index] = activity_encoding

                pos_index += 1

        return features.view((2*6*8*8))

    def label_to_tensor(label):
        label_t = torch.zeros((3))

        label_t[ColorBased.label_dict[label]] = 1.

        return label_t
