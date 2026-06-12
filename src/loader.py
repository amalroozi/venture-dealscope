import pandas as pd
import numpy as np


def load_data(path='data/startups_clean.csv'):
    data = pd.read_csv(path)
    return data