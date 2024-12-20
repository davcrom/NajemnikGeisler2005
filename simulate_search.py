import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import psyfun
import util

# Load the psychometric detection data
fname_psy = 'data/humans/detection_aggregate.csv'  # file from psychopy detection experiment
df_psy = util.load_psychometric_data(fname_psy)

# Create a visibility map object
vismap = psyfun.VisibilityMap(df_psy, log_contrast=True)  
# Fit a psychometric function for each eccentricity 
vismap.fit_psyfuns()
# Fit a linear model to predict what the psychometric function for any eccentricity in the tested range
vismap.interpolate_parameters()
# Create a bayesian optimal searcher with the vismap object
searcher = psyfun.BayesianOptimalSearcher(vismap, criterion=0.99)

# Load the experimental conditions
df_conditions = pd.read_excel('conditions_search.xlsx')
# Set a number of repetitions for each condition
n_reps = 50
n_trials = n_reps * len(df_conditions)
# Create a dataframe of trials with n_conditions * n_reps
df_trials = pd.DataFrame(np.tile(df_conditions['grating_eccentricity'], n_reps), columns=['eccentricity'])
# Assign a random radial position to the target for each trial
df_trials['angle'] = np.random.choice(
    np.arange(-np.pi, np.pi, 2 * np.pi / 32),
    size=n_trials
    )
# Convert the radial position to x,y coordinates
df_trials['x'] = df_trials.apply(lambda x: np.cos(x['angle']) * x['eccentricity'], axis='columns')
df_trials['y'] = df_trials.apply(lambda x: np.sin(x['angle']) * x['eccentricity'], axis='columns')
# Assign a costant contrast
contrast = np.log10(df_conditions['grating_contrast'].unique()[0])

# Run a search for each trial we just created
search_data = []
for i, trial in df_trials.iterrows():
    print(f'Trial no. {i+1}, eccentricity {trial["eccentricity"]}')
    searcher.initialize_trial(trial['x'], trial['y'], contrast)
    searcher.search(plot=False)
    search_data.append(searcher.convert_to_eyedata_series())
df_search = pd.DataFrame(search_data)

# Get the total number of fixations and the saccade lengths for each trial
df_search['n_fixations'] = df_search['fixations'].apply(len)
df_search['saccade_lengths'] = df_search.apply(util.get_saccade_lengths, axis='columns')

# Save the resulting data
df_search.to_pickle('data/models/bayesian_optimal_01.pkl')