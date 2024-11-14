
"""
create hierarchical_model zip file with instantiate in it
the zip file should be standalone package
uses hiderachicial_model.yaml to construct zip file
"""

from pathlib import Path
import git
import sqlite3
import pandas as pd
from flatten_dict import flatten
from shared_utils import utils
import shutil
import yaml

def get_model_name_from_tree(node):
    """
    extract only model name from tree from hierarchical model.yaml
    """
    if type(node) is int: return []
    
    model_names = [node['path']]
    for child in node['out'].values(): 
        model_names += get_model_name_from_tree(child)
        
    return model_names

if __name__ == "__main__":
    # run as python -m hierarchical_model.pipeline

    hierarchical_yaml_Path = Path('./hierarchical_model/hierarchical_model.yaml')
    trained_path = Path('/data/raspy/trained_models') # where to look to grab trained models
    save_path = Path('/data/raspy/trained_models') # where to store the hierarhical model
    instantiate_path = Path('./hierarchical_model/instantiate.py') # where is the hiearchicla model instantiate


    # Write into sql dataset
    train_on_server = True
    repo = git.Repo(search_parent_directories=True)
    conn = sqlite3.connect('/data/raspy/sql/sql_eeg.db') if train_on_server else None
    # TODO: lack of adding this record into sql table

    # Generate model name
    model_name = utils.model_namer(conn, train_on_server, 'Hierarchical')
    save_path = save_path / model_name
    save_path.mkdir(exist_ok=True)
    print("Model Name:", model_name)

    # move everything into a directory inside trained_path
    with open(hierarchical_yaml_Path, 'r') as file: h_yaml = yaml.safe_load(file)
    model_names = get_model_name_from_tree(h_yaml['hierarchical_model'])
    shutil.copy(hierarchical_yaml_Path, save_path)
    shutil.copy(instantiate_path, save_path / 'instantiate.py')
    for model_name in model_names: 
        shutil.copytree(trained_path / model_name, save_path / model_name)

