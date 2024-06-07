import json
import os
import numpy as np
from tqdm import tqdm

def find_deepest_eval(eval_list):
    max_depth = 0
    eval = None

    for x in eval_list:
        depth = x['depth']
        
        if depth > max_depth:
            max_depth = depth
            eval = x['pvs']

    return eval 

def categorize(eval_list):
    # select first pvs for simplicity
    eval = eval_list[0]

    if 'mate' in eval:
        return np.sign(eval['mate'])

    if eval['cp'] > 150:
        return 1
    elif eval['cp'] < -150:
        return -1
    else:
        return 0

dir_path = f'{os.getcwd()}/data'
lichess_db_path = f'{dir_path}/lichess_db_eval.jsonl'

# unpack + read db
if not os.path.isfile(lichess_db_path):
    os.system(f'unzstd {lichess_db_path}.zst')

with open(lichess_db_path, 'r') as json_file:
    lichess_db = list(json_file)

# for every position, grab deepest eval and categorize (white > 150 > draw > -150 > black)
white_wins = []
black_wins = []
draws = []

for line in tqdm(lichess_db):
    x = json.loads(line)
    fen = x['fen']

    eval = find_deepest_eval(x['evals'])
    score = categorize(eval)

    if score == 1:
        white_wins.append(fen)
    elif score == -1:
        black_wins.append(fen)
    else:
        draws.append(fen)

def write_to_file(path, l):
    with open(path, "w") as file:
        file.write("\n".join(l))

# for training:
#           imbalance in game outcomes is dealt with during sampeling
# for testing:
#           equal number of game outcomes is taken into testing db (0.2 * smallest db)

test_pct = 0.2
smallest_len = min(len(white_wins), len(black_wins), len(draws))

test_size = round(smallest_len * test_pct)

write_to_file(f'{dir_path}/test/white.txt', white_wins[:test_size])
write_to_file(f'{dir_path}/test/black.txt', black_wins[:test_size])
write_to_file(f'{dir_path}/test/draw.txt', draws[:test_size])

write_to_file(f"{dir_path}/train/white.txt", white_wins[test_size:])
write_to_file(f"{dir_path}/train/black.txt", black_wins[test_size:])
write_to_file(f"{dir_path}/train/draw.txt", draws[test_size:])