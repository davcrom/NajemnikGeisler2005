import ast
import re
import numpy as np
import pandas as pd

df = pd.read_csv('data/models/bayesian_optimal_00.csv')

def parse_string(string):
    return np.array(ast.literal_eval('[[' + '],['.join([s.lstrip(' ').rstrip(' ').replace(' ', ',') for s in re.sub(r'\s+', ' ', string.lstrip('[[').rstrip(']]').replace('\n ',',')).split('],[')]) + ']]'))
    
df['fixations'] = df['fixations'].apply(parse_string)

df.to_pickle('data/models/bayesian_optimal_00.pkl')