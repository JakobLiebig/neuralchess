import json
import pandas as pd
from tqdm import trange

def find_deepest_eval(eval_list):
    max_depth = 0
    eval = None

    for x in eval_list:
        depth = x['depth']
        
        if depth > max_depth:
            max_depth = depth
            eval = x['pvs']

    return eval, max_depth

def pvs_list_to_score(eval_list):
    # select first pvs for simplicity
    eval = eval_list[0]

    if 'mate' in eval:
        return 'Nan', eval['mate']

    return eval['cp'], 'Nan'

def get_prop_winner(cp, mate):
    if mate != 'Nan':
        if int(mate) > 0:
            return 'White'
        else:
            return 'Black'
    else:
        if int(cp) > 150:
            return 'White'
        elif int(cp) < -150:
            return 'Black'
        else:
            return 'Draw'

def get_turn_eval(active_color, prop_winner):
    if prop_winner == 'Draw':
        turn_eval = 'Draw'
    elif (prop_winner == 'White') == active_color:
        turn_eval = 'Win'
    else:
        turn_eval = 'Loss'
    
    return turn_eval

src_path = 'nn/data/lichess/src/lichess_db_eval.jsonl'
out_path = 'nn/data/lichess/data.csv'

# the eval db consists of 41113403 positions, thats way too many
# i will only grab a subset
len_subset = 20000000

df_data = []

with open(src_path, 'r') as source_jsonl:
    for i in trange(len_subset):
        json_line = json.loads(source_jsonl.readline())
        
        fen = json_line['fen']
        pvs, depth = find_deepest_eval(json_line['evals'])
        cp, mate = pvs_list_to_score(pvs)
        prop_winner = get_prop_winner(cp, mate)
        
        active_color = fen.split(' ')[1] == 'w'
        turn_eval = get_turn_eval(active_color, prop_winner)
        
        df_data.append({'FEN' : fen, 'Depth' : depth, 'CP' : cp, 'Mate' : mate, 'PropableWinner' : prop_winner ,'TurnEvaluation' : turn_eval})
    
df = pd.DataFrame.from_dict(data=df_data)
df.to_csv(out_path, index=False)