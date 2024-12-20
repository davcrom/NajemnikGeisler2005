import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import psyfun
import util

fnames = ['DC_search_2024-06-11_22h24.37.236', 'BJ_search_2024-11-02_16h55.49.047', 'DC_search_2024-11-03_17h13.20.548']
display_dims = [(51, 29, -1), (61, 35, 1), (61, 35, 1)]

df_agg = pd.DataFrame()
for (w, h, x), fname in zip(display_dims, fnames):
    # Load the visual search data
    fname_search = f'data/humans/{fname}.csv'  # file from psychopy search experiment
    fname_trials = f'data/humans/{fname}_events.csv'  # file with trial timestamps from eye tracker
    df_search = util.load_search_data(fname_search, fname_trials)
    
    # Plot trial durations and optionally filter out trials that took too long
    rt_threshold = 20  # set a threshold here (seconds)
    df_search = df_search.query(f'trial_duration < {rt_threshold * 10**9}')
    
    # Load the eye tracking data
    fname_gaze = f'data/humans/{fname}_gaze.csv'  # file with pupil position data mapped on to the screen
    df_fixations = util.load_fixation_data(fname_gaze)  # here, we only consider fixations
    
    # Check the distribution of starting fixations to see if there is an offset
    xx, yy, (x_offset, y_offset) = util.get_starting_fixations(df_search, df_fixations)
    # Correct all gaze coordinates
    df_fixations = util.correct_offset(df_fixations, (x_offset, y_offset))
    
    # Convert postitions to degrees visual angle
    df_fixations = util.pos2deg(df_fixations, w, h, x)
    
    # Get the fixations for each trial
    df_search['fixations'] = df_search.apply(util.get_search_fixations, df_fix=df_fixations, axis='columns')  
    df_search['n_fixations'] = df_search['fixations'].apply(len)
    df_search['saccade_lengths'] = df_search.apply(util.get_saccade_lengths, axis='columns')
    
    # Filter out trials where too many off-screen fixations were detected
    n_trials = len(df_search)
    df_search = df_search[df_search['fixations'].apply(lambda x: np.isnan(x).mean() < 0.25)]
    print(f'{n_trials - len(df_search)} of {n_trials} trials removed')

    df_agg = pd.concat([df_agg, df_search])
df_agg.reset_index(inplace=True)
df_agg.to_pickle('data/humans/search_aggregate.pkl')