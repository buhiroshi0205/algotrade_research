import os

def make_directories(experiment_name):
  os.makedirs(f'../models/{experiment_name}', exist_ok=True)