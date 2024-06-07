import wget
import os

url = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
file_path = f"{os.getcwd()}/data/lichess_db_eval.jsonl.zst"

wget.download(url, file_path)