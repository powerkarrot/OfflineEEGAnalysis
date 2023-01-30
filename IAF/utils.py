
from itertools import chain, repeat
import os
from Settings import *
import pandas as pd
import mne
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import compress
import philistine
import autoreject
from autoreject import get_rejection_threshold

def make_raw(dir, preprocess=True):

    dfEEG = pd.read_csv(f"{data_path}{dir}.csv")
    dfEEG.drop(["TimeLsl", "Time"], axis=1, inplace=True)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage('standard_1020',  match_case=False)
    samples = dfEEG.T
    raw = mne.io.RawArray(samples, info)
    
    if preprocess:
        raw = preprocess_raw(raw)
    return raw

def make_epoch(raw):
    
    epochs = mne.make_fixed_length_epochs(raw.copy(), preload=True, duration = 1.)
    
    reject = get_rejection_threshold(epochs, ch_types = 'eeg', verbose=False)      
    print("The rejection dictionary is %s " %reject)
    epochs.drop_bad(reject=reject)
    
    return  epochs

def preprocess_raw(raw):
    raw.notch_filter(60., n_jobs='cuda')       
    raw.filter(1., 70., None, fir_design='firwin', n_jobs='cuda')
    raw.set_eeg_reference('average', projection=True)
    return raw

def select_channels_picks(instance, picked_channels):
    mask = np.array(np.isin(channels, picked_channels, invert=True), dtype = bool)
    excl = list(compress(channels, mask))
    return mne.pick_types(instance.info, eeg=True,  exclude=excl)


def test_channels(raw1, raw2, channel_list):
    bad_chs = []
    for ch in enumerate(channel_list):
        p =  select_channels_picks(raw1, ch)
        try:
            philistine.mne.attenuation_iaf([raw1,raw2], picks=p, savgol='diff', resolution=.1)
        except Exception as e:
            print(e)
            bad_chs.append(ch)
    return(bad_chs)
            
def test_channels2(raw1,channel_list):
    bad_chs = []
    for ch in enumerate(channel_list):
        p =  select_channels_picks(raw1, ch)
        try:
            philistine.mne.savgol_iaf(raw1, picks=p)
        except:
            bad_chs.append(ch)
    
    return(bad_chs)