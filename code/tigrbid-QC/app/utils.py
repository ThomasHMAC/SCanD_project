import json
import pandas as pd
from glob import glob

def load_pipeline_config(json_file="config.json"):
    with open(json_file, "r") as f:
        return json.load(f)

def load_metrics(pipeline_name, config):
    pipeline = config[pipeline_name]
    df = pd.read_csv(pipeline["metrics_file"], sep="\t").set_index("Subject")
    return df

def load_visuals(pipeline_name, config):
    pipeline = config[pipeline_name]
    return sorted(glob(pipeline["visual_glob"]))