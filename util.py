import numpy as np
import pandas as pd

from scipy import stats
from scipy import optimize
from scipy import integrate

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

ITI = 1.5 * 10**9
MONITOR_WIDTH = 51  # cm
MONITOR_HEIGHT = 29
VIEWING_DISTANCE = 57

NOISE_RADIUS = np.arctan2((MONITOR_HEIGHT / 2), VIEWING_DISTANCE) * 180 / np.pi
TARGET_RADIUS = 0.5

def load_psychometric_data(fname):
    df = pd.read_csv(fname)
    df = df.dropna(subset='grating_contrast')  # remove rows corresponding to non-trial routines
    df['trial'] = df['detection_trials.thisN'].astype(int) + 1
    df = df.set_index('trial')
    df['correct'] = df['correct_key'] == df['key_resp.keys']  # mark trials as correct
    
    columns = ['grating_contrast', 'grating_eccentricity', 'correct']
    return df[columns]

def load_search_data(fname_search, fname_trials):
    df_search = pd.read_csv(fname_search)
    df_search = df_search.dropna(subset='grating_contrast')  # remove rows corresponding to non-trial routines
    df_search['trial'] = df_search['search_trials.thisN'].astype(int) + 1
    df_search = df_search.set_index('trial')
    
    df_trials = pd.read_csv(fname_trials)
    df_trials = df_trials.iloc[1:]  # remove the recording start timestamp
    trial_duration = np.diff(df_trials['timestamp [ns]'])  # compute trial durations
    df_trials = df_trials.iloc[:-1]  # remove recording stop timestamp
    df_trials.index.name = 'trial'
    df_trials['fixation'] = df_trials['timestamp [ns]']       # start of fixation
    df_trials['trial_start'] = df_trials['fixation'] + ITI   # start of first interval
    df_trials['trial_stop'] = df_trials['fixation'] + trial_duration  # trial stop time
    df_trials['trial_duration'] = trial_duration
    
    columns_search = ['grating_eccentricity', 'grating_angle']
    columns_trials = ['fixation', 'trial_start', 'trial_stop', 'trial_duration']
    return pd.concat([df_search[columns_search], df_trials[columns_trials]], axis=1)

def load_fixation_data(fname_gaze):
    df = pd.read_csv(fname_gaze)
    df = df.rename(lambda x: x.replace(' ', '_'), axis='columns')  # remove spaces from column names
    # Rename the important columns to something more convenient
    df = df.rename(columns={
        'gaze_position_on_surface_x_[normalized]':'x_pos',
        'gaze_position_on_surface_y_[normalized]':'y_pos',
        'timestamp_[ns]':'timestamp'
        })
    df_fix = df.dropna(subset='fixation_id').query('gaze_detected_on_surface == True')
    columns = ['timestamp', 'x_pos', 'y_pos', 'fixation_id']
    return df[columns]

def get_starting_fixations(df_trials, df_fix):
    xx = np.full(len(df_trials), np.nan)
    yy = np.full(len(df_trials), np.nan)
    for i, (idx, trial) in enumerate(df_trials.iterrows()):
        t0 = trial['fixation']
        t1 = trial['trial_start']
        df_fixepoch = df_fix.query(f'(timestamp > {t0}) & (timestamp < {t1})')
        xx[i] = np.median(df_fixepoch['x_pos'].values)
        yy[i] = np.median(df_fixepoch['y_pos'].values)
    return xx, yy, (np.median(xx), np.median(yy))

def plot_starting_fixations(df_trials, df_fix, ax=None):
    if ax is None:
        fig, ax = plt.subplots()    
    xx, yy, (x_offset, y_offset) = get_starting_fixations(df_trials, df_fix)
    ax.hist2d(xx, yy, bins=np.linspace(0, 1, 51))
    ax.scatter(x_offset, y_offset, marker='x', color='black')
    ax.axvline(0.5, ls='--', color='gray')
    ax.axhline(0.5, ls='--', color='gray') 
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return ax, (x_offset, y_offset)

def correct_offset(df, offset):
    ## TODO: don't return df, just columns
    x_offset, y_offset = offset
    df['x_pos'] = df['x_pos'] - x_offset + 0.5
    df['y_pos'] = df['y_pos'] - y_offset + 0.5
    return df

def pos2deg(df, display_width=MONITOR_WIDTH, display_height=MONITOR_HEIGHT, x_flip=-1):
    df['x_pos_cm'] = (df['x_pos'] * display_width) - (display_width / 2)  
    df['x_deg'] = np.arctan2(x_flip * df['x_pos_cm'], VIEWING_DISTANCE) * 180 / np.pi
    df['y_pos_cm'] = (df['y_pos'] * display_height) - (display_height / 2)  
    df['y_deg'] = np.arctan2(-1 * df['y_pos_cm'], VIEWING_DISTANCE) * 180 / np.pi
    return df

def _plot_search_trial_old(trial, df_trials, df_fix, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot stimulus elements 
    noise_circle = Circle((0, 0), NOISE_RADIUS, lw=2, fc='none', ec='black')
    ax.add_patch(noise_circle)
    eccentricity = df_trials.loc[trial]['grating_eccentricity']
    angle = df_trials.loc[trial]['grating_angle']
    target_x = eccentricity * np.cos(angle)
    target_y = eccentricity * np.sin(angle)
    target_circle = Circle((target_x, target_y), TARGET_RADIUS * 2, lw=2, fc='none', ec='C0')
    ax.add_patch(Circle((target_x, target_y), TARGET_RADIUS, lw=2, fc='none', ec='C0'))
    ax.add_patch(Circle((target_x, target_y), TARGET_RADIUS * 3, lw=2, ls='--', fc='none', ec='C0'))
    max_coord = np.ceil(NOISE_RADIUS)
    ax.set_xlim([-max_coord, max_coord])
    ax.set_ylim([-max_coord, max_coord])

    # Get trial data
    t0 = df_trials.loc[trial]['trial_start'] - 0.5 * 10**9
    t1 = df_trials.loc[trial]['trial_stop']
    df_searchepoch = df_fix.query(f'(timestamp > {t0}) & (timestamp < {t1})')
    # Get median position for each detected fixation
    def get_median(fix):
        x = np.median(fix['x_deg'])
        y = np.median(fix['y_deg'])
        e = np.sqrt(x**2 + y**2)
        return (x, y) if e <= NOISE_RADIUS else (np.nan, np.nan)
    fixations = df_searchepoch.groupby('fixation_id').apply(get_median)

    # Plot saccades
    xx, yy = np.row_stack([fix for fix in fixations.values]).T
    ax.plot(xx, yy, marker='o', color='gray')
    for i, (x, y) in enumerate(zip(xx, yy)):
        if np.isnan(x): continue
        ax.text(x + 1, y + 1, f'{i + 1}', fontsize=12, ha='center', va='center')

    return ax, fixations

def plot_search_trial(trial, df_search, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot stimulus elements 
    noise_circle = Circle((0, 0), NOISE_RADIUS, lw=2, fc='none', ec='black')
    ax.add_patch(noise_circle)

    trial = df_search.loc[trial]
    eccentricity = trial['grating_eccentricity']
    angle = trial['grating_angle']
    target_x = eccentricity * np.cos(angle)
    target_y = eccentricity * np.sin(angle)
    target_circle = Circle((target_x, target_y), TARGET_RADIUS * 2, lw=2, fc='none', ec='C0')
    ax.add_patch(Circle((target_x, target_y), TARGET_RADIUS, lw=2, fc='none', ec='C0'))
    ax.add_patch(Circle((target_x, target_y), TARGET_RADIUS * 3, lw=2, ls='--', fc='none', ec='C0'))
    max_coord = np.ceil(NOISE_RADIUS)
    ax.set_xlim([-max_coord, max_coord])
    ax.set_ylim([-max_coord, max_coord])

    # Plot saccades
    xx, yy = np.row_stack([fix for fix in trial['fixations']]).T
    ax.plot(xx, yy, marker='o', color='gray')
    for i, (x, y) in enumerate(zip(xx, yy)):
        if np.isnan(x): continue
        ax.text(x + 1, y + 1, f'{i + 1}', fontsize=12, ha='center', va='center')

    return ax

def get_search_fixations(trial, df_fix):
    t0 = trial['trial_start'] - 0.5 * 10**9
    t1 = trial['trial_stop']
    df_searchepoch = df_fix.query(f'(timestamp > {t0}) & (timestamp < {t1})')
    # Get median position for each detected fixation
    def get_median(fix):
        x = np.median(fix['x_deg'])
        y = np.median(fix['y_deg'])
        e = np.sqrt(x**2 + y**2)
        return (x, y) if e <= NOISE_RADIUS else (np.nan, np.nan)
    fixations = df_searchepoch.groupby('fixation_id').apply(get_median)
    return np.row_stack([fix for fix in fixations.values])

def plot_fixation_distribution(df_search, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    noise_circle = Circle((0, 0), NOISE_RADIUS, lw=2, fc='none', ec='white', zorder=2)
    ax.add_patch(noise_circle)
    max_coord = np.ceil(NOISE_RADIUS)
    fixations = np.row_stack([fix['fixations'][1:] for idx, fix in df_search.iterrows()])
    ax.hist2d(*fixations.T, bins=np.linspace(-max_coord, max_coord, 16))
    return ax

def get_saccade_lengths(trial):
    return np.sqrt((np.diff(trial['fixations'], axis=0)**2).sum(axis=1))

    