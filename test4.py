# %%
import numpy as np
import mne
from mne.time_frequency import psd_multitaper
import pandas as pd
import tqdm 
import matplotlib.pyplot as plt

# Import some NeuroDSP functions to use with MNE
from neurodsp.spectral import compute_spectrum, trim_spectrum
from neurodsp.burst import detect_bursts_dual_threshold
from neurodsp.rhythm import compute_lagged_coherence

# Import NeuroDSP plotting functions
from neurodsp.plts import (plot_time_series, plot_power_spectra,
                           plot_bursts, plot_lagged_coherence)

import os
from pathlib import Path

# %%
lstPIds = []
path = "./Data/"
for filename in os.listdir(path):
    if filename.endswith(".csv"): 
        lstPIds.append(int(filename.split("-")[0].replace("ID", "")))
    else:
        continue
lstPIds = list(set(lstPIds))
print(lstPIds)

# %%
#data = np.loadtxt('./dfEEG.csv', delimiter=',').readlines()
data = pd.read_csv('./dfEEG.csv', delimiter=',')

path = "./Data/"


# %%
for pid in tqdm.tqdm(lstPIds):
    
    dfState = pd.read_csv(f"{path}ID{pid}-state.csv")
    dfState = pd.read_csv(f"{path}ID{pid}-state.csv")

    dfStart = dfState[dfState.State == "start"].copy()
    dfEnd = dfState[dfState.State == "end"][["Time"]].iloc[:len(dfStart)]
    dfStart = dfStart.rename(columns={"Time":"TimeStart"})
    dfStart.TimeStart = dfStart.TimeStart #+ 60
    dfStart["TimeEnd"] = dfEnd.Time.values
    del dfStart["State"]
    dfStart["Duration"] = dfStart.TimeEnd - dfStart.TimeStart
    df = dfStart[dfStart.BlockNumber != -2].copy()
        
    dfEEG = pd.read_csv(f"{path}ID{pid}-EEG.csv")
    dfEEG = dfEEG.rename(columns={"Value0": "F3", "Value1": "C3", "Value2": "P3", "Value3": "P4", "Value4": "C4", "Value5": "F4", "Value6": "Pz"})
    #dfEEG.drop(['Value7'], axis=1)
    #dfEEG.drop("Time", axis =1, inplace=True)
    dfEEG.drop("TimeLsl", axis =1, inplace=True)

   
    #dfEEG["Value7"] = dfEEG.Time-dfState.Time.iloc[0]
    #dfEEG.drop(['Value7'], axis=1)
    #dfEEG.drop("Time", axis =1, inplace=True)
    ch_names = ['Time', 'F3','C3','P3','P4','C4','F4','Pz', 'Value7']
    ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',  'misc']
    #ch_types = ["eeg"]*dfEEG.shape[1]
    
    sfreq=300
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage('standard_1020',  match_case=False)

    samples = dfEEG.T*1e-6

    raw = mne.io.RawArray(samples, info)
    print("pid:", pid)
    raw.plot(scalings=150e-4, n_channels=7)
    
    dfEEG.plot(x="Time", y=["F3", "C3","P3","P4","C4","F4","Pz"], figsize=(15, 8))

    
    raw.notch_filter(50)
    raw.plot_psd()
    
    ica = mne.preprocessing.ICA(n_components=7, random_state=97, max_iter=800)
    ica.fit(raw)
    ica.exclude = [1, 2]  # details on how we picked these are omitted here
    ica.plot_properties(raw, picks=ica.exclude)
    
    raw.filter(l_freq = 1, h_freq = 60)
    raw.plot_psd(0.5, 40, 0,360)
    
    raw.filter(l_freq = 7, h_freq = 12)
    raw.plot_psd()
    
    #inst_new = raw.copy().filter(300, l_freq=7.5, h_freq=12.5)
    
    #mne.filter.filter_data(inst_new, 300, l_freq=7.5, h_freq=12.5)  







# %%
for pid in tqdm.tqdm(lstPIds):
    
    if(pid > 1):
        break
    
    dfState = pd.read_csv(f"{path}ID{pid}-state.csv")
    dfState = pd.read_csv(f"{path}ID{pid}-state.csv")

    # dfStart = dfState[dfState.State == "start"].copy()
    # dfEnd = dfState[dfState.State == "end"][["Time"]].iloc[:len(dfStart)]
    # dfStart = dfStart.rename(columns={"Time":"TimeStart"})
    # dfStart.TimeStart = dfStart.TimeStart #+ 60
    # dfStart["TimeEnd"] = dfEnd.Time.values
    # del dfStart["State"]
    # dfStart["Duration"] = dfStart.TimeEnd - dfStart.TimeStart
    # df = dfStart[dfStart.BlockNumber != -2].copy()
        
    dfEEG = pd.read_csv(f"{path}ID{pid}-EEG.csv")
    dfEEG = dfEEG.rename(columns={"Value0": "F3", "Value1": "C3", "Value2": "P3", "Value3": "P4", "Value4": "C4", "Value5": "F4", "Value6": "Pz"})
    #dfEEG.drop(['Value7'], axis=1)
    #dfEEG.drop("Time", axis =1, inplace=True)
    dfEEG.drop("TimeLsl", axis =1, inplace=True)

    ch_names = ['Time', 'F3','C3','P3','P4','C4','F4','Pz', 'Value7']
    ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',  'misc']
    
    sfreq=250
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage('standard_1020',  match_case=False)

    samples = dfEEG.T*1e-6

    raw = mne.io.RawArray(samples, info)
    print("pid:", pid)
    raw.filter(l_freq=0.1, h_freq=30)
    #raw.plot( scalings='20e-4')


    dstate = pd.read_csv(f"{path}ID{pid}-state.csv")
    
    dfAll = pd.merge(dfEEG, dstate, on =["Time"], how="outer")#.fillna(method='ffill')
    #dfAll["Time"] = pd.to_datetime(dfAll["Time"],origin='unix',unit='s')
    dfAll = dfAll.sort_values(by="Time")
    #dfAll["Time"] =  dfAll['Time'].apply(lambda x: x.timestamp())

    
    dfAll = dfAll.drop(columns=["Value7","AdaptationStatus", "NBackN", "State"] )
    dfAll.fillna(method='ffill', inplace=True)
    dfAll = dfAll.drop(dfAll[dfAll.BlockNumber < 0].index)
    dfAll = dfAll.dropna()
    
    ch_names = ['Time', 'F3','C3','P3','P4','C4','F4','Pz', 'BlockNumber']
    ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',  'misc']
    sfreq=250
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage('standard_1020',  match_case=False)
    samples = dfAll.T*1e-6
    raw = mne.io.RawArray(samples, info)
    
    #notch filter interference at 50 Hz, not sure its needed though?
    raw.notch_filter(50)
    #bandpass filter, cut signal outside the EEG spectrum
    raw.filter(l_freq = 1, h_freq = 60)
    #raw.filter(l_freq=0.1, h_freq=40)
    raw.plot( scalings='20e-4', n_channels = 7, duration=10)


    # raw.notch_filter(50)
    raw.plot_psd(area_mode='range', show=False, average=True)
    raw.plot_psd()
    raw.plot_psd_topo()
    
    # ica = mne.preprocessing.ICA(n_components=7, random_state=97, max_iter=800)
    # ica.fit(raw)
    # ica.exclude = [1, 2]  # details on how we picked these are omitted here
    # ica.plot_properties(raw, picks=ica.exclude)
    # raw.plot( scalings='20e-4', n_channels = 7)

    #dfBlocks = np.array()
    for x in range(1, 7):
        data = dfAll.loc[dfAll['BlockNumber'] == x]
        df = pd.DataFrame(data)
        #dfBlocks.append(df)
        data.plot(x="Time", y=["F3", "C3","P3","P3","C4","F4","Pz"])

        ch_names = ['Time', 'F3','C3','P3','P4','C4','F4','Pz', 'BlockNumber']
        ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',  'misc']
        
        sfreq=250
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        info.set_montage('standard_1020',  match_case=False)

        samples = df.T*1e-6

        raw = mne.io.RawArray(samples, info)
        raw_copy = raw.copy()
        print("pid:", pid)
        
        raw.set_eeg_reference('average', projection=True)
        #raw.set_eeg_reference(ref_channels=['Pz'])

        
        # create the annotations
        # rem_annot = mne.Annotations(onset=[0, 360, 720, 1080, 1440, 1800, 2160],
        #                             duration=[360]*7,
        #                             description=['Blocks'] * 7)
        # raw.set_annotations(rem_annot)
        # (rem_events,rem_event_dict) = mne.events_from_annotations(raw)
        
        #plt.figure()
        #ax = plt.axes()
        #ax.set_title("comparison")
                        
        #notch filter 50Hz interference
        raw.notch_filter(50)
        raw.plot_psd()
        # filter out alpha
        raw.filter(l_freq=7.5, h_freq=12.5)

        #plot
        raw.plot( scalings='20e-5', timeformat='s', duration=3600)
        raw.plot_psd_topo()
        raw.plot_psd()
        
        raw.plot_projs_topomap(colorbar=True)

        epochs = mne.make_fixed_length_epochs(raw, preload=False)
        epochs.load_data().filter(l_freq=8, h_freq=12)
        alpha_data = epochs.get_data() 
        
        
        bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),(12, 30, 'Beta'), (30, 45, 'Gamma')]
        epochs.plot_psd_topomap(bands=bands, normalize=True)

        
        # independent component analysis (ICA)
        ica = mne.preprocessing.ICA(n_components=7, random_state=97, max_iter=800)
        ica.fit(raw)
        ica.exclude = [1, 2]  # details on how we picked these are omitted here
        ica.plot_properties(raw, picks=ica.exclude)
        
        # Compute the power spectral density (PSD) using multitapers.
        f, ax = plt.subplots()
        picks = mne.pick_types(raw_copy.info, meg=False, eeg=True, eog=False,
                       stim=False)
        psds, freqs = psd_multitaper(raw_copy, low_bias=True,
                             fmin=7.5, fmax=12.5, proj=False, picks=picks,
                             n_jobs=1)

        psds = 10 * np.log10(psds)
        psds_mean = psds.mean(0)
        psds_std = psds.std(0)
        
        #peak power at freq
        #peak_cf = freqs[np.argmax(powers)]
        
        #avg alpha power? is it like this? no idea.
        avg_cf = np.mean(psds)
        
        #print(peak_cf)
        print(avg_cf)
        print("multitaper")
        

        ax.plot(freqs, psds_mean, color='k')
        ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                        color='k', alpha=.5)
        
        ax.set(title='Multitaper PSD', xlabel='Frequency',
        ylabel='Power Spectral Density (dB)')
        plt.show()
        ch_label = 'Pz'
        #other channels too -.-
        sig, times = raw.get_data(mne.pick_channels(raw.ch_names, [ch_label]),return_times=True)
        
        sig = np.squeeze(sig)
        
        fs = raw.info['sfreq']
        
        # Calculate the power spectrum, using median Welch's & extract a frequency range of interest
        freqs, powers = compute_spectrum(sig, fs, method='welch', avg_type='median')
        freqs, powers = trim_spectrum(freqs, powers, [7.5, 12.5])
        
        #peak power at freq
        #peak_cf = freqs[np.argmax(powers)]
        
        #avg alpha power
        avg_cf = np.mean(powers)
        
        #print(peak_cf)
        print(avg_cf)
        print("the other one")

        
        plot_power_spectra(freqs, powers)
        plt.plot(freqs[np.argmax(powers)], np.max(powers), '.r', ms=12)
        
        
        
        
        
        
    #spectrum = epochs['visual/right'].compute_psd()
    #spectrum.plot_topomap()

    #raw.plot(events=rem_events, scalings='20e-3')




# %%
for df in dfBlocks:
  
    ch_names = ['Time', 'F3','C3','P3','P4','C4','F4','Pz', 'BlockNumber']
    ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',  'misc']
    
    sfreq=250
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage('standard_1020',  match_case=False)

    samples = df.T*1e-6

    raw = mne.io.RawArray(samples, info)
    print("pid:", pid)
    
    # create the annotations
    # rem_annot = mne.Annotations(onset=[0, 360, 720, 1080, 1440, 1800, 2160],
    #                             duration=[360]*7,
    #                             description=['Blocks'] * 7)
    # raw.set_annotations(rem_annot)
    # (rem_events,rem_event_dict) = mne.events_from_annotations(raw)
    
    #raw.filter(l_freq=0.1, h_freq=30)
    #raw.plot(start=60)
    #raw.plot( scalings='20e-3')
    
    
    #spectrum = epochs['visual/right'].compute_psd()
    #spectrum.plot_topomap()

    #raw.plot(events=rem_events, scalings='20e-3')



# %%
data.plot(x="TimeNorm", y=["F3", "C3","P3","P3","C4","F4","Pz"])


# %%
original_raw = raw.copy()
original_raw.load_data()
rereferenced_raw, ref_data = mne.set_eeg_reference(original_raw, ['Pz'],
                                                   copy=True)
fig_orig = original_raw.plot()
fig_reref = rereferenced_raw.plot()

#raw.plot(n_channels=7)
raw.plot(scalings=3e4, n_channels=6, duration=6)




