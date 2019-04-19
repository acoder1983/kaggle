import pandas as pd
def read_data(file):
    return pd.read_table('raw_data/test.txt',header=None)