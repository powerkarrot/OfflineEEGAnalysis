# %%
import numpy as np
import mne
from mne.time_frequency import psd_multitaper, psd_welch
import pandas as pd
import tqdm 
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from autoreject import get_rejection_threshold

# composite Simpson's rule
from scipy.integrate import simpson

# Import some NeuroDSP functions to use with MNE
from neurodsp.spectral import trim_spectrum


# FOOOF imports
from fooof.bands import Bands

import os

# %%
mne.set_log_level(False)
mne.utils.set_config('MNE_USE_CUDA', 'true')  
plt.rcParams.update({'figure.max_open_warning': 0})


# %%
channel_groups =[['F3', 'F4'],['F3', 'F4', 'C3', 'C4'],['P3', 'Pz', 'P4'],['F3','C3','P3','P4','C4','F4','Pz']]
ch_names = ['Time', 'F3','C3','P3','P4','C4','F4','Pz', 'BlockNumber']
ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',  'misc']

bands = Bands({'theta': [4, 8], 'alpha': [8, 12]})   
 
plot_plots = True       
save_plots = False
draw_plots = True

# %%

# %%
# bad channels
# TODO fill for all participants :')
bads = [[['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [[], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []],
        [['F2', 'F3'], [], ['C4'], [], [], [],  []]
]

# %%
pws_lst = list()

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

for pid in tqdm.tqdm(lstPIds):
    
    if (pid != 5):
       continue
    # if (pid > 1):
    #         break
    print("pid:", pid)

    dfState = pd.read_csv(f"{path}ID{pid}-state.csv")
    dfState = pd.read_csv(f"{path}ID{pid}-state.csv")
        
    dfEEG = pd.read_csv(f"{path}ID{pid}-EEG.csv")
    dfEEG = dfEEG.rename(columns={"Value0": "F3", "Value1": "C3", "Value2": "P3", "Value3": "P4", "Value4": "C4", "Value5": "F4", "Value6": "Pz"})
    dfEEG.drop("TimeLsl", axis =1, inplace=True)

    dstate = pd.read_csv(f"{path}ID{pid}-state.csv")
    
    dfAll = pd.merge(dfEEG, dstate, on =["Time"], how="outer")#.fillna(method='ffill')
    dfAll = dfAll.sort_values(by="Time") # inplace?
    
    dfAll = dfAll.drop(columns=["Value7","AdaptationStatus", "NBackN", "State"] )
    dfAll.fillna(method='ffill', inplace=True)
    dfAll = dfAll.drop(dfAll[dfAll.BlockNumber < 0].index)
    dfAll = dfAll.dropna()
    

    for x in range(1, 2):  
        
        # if(x > 1):
        #     break
        
        # Prepare data 
        # region
        data = dfAll.loc[dfAll['BlockNumber'] == x]
        df = pd.DataFrame(data)
        # data.plot(x="Time", y=["F3", "C3","P3","P3","C4","F4","Pz"])

        sfreq=250
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        info.set_montage('standard_1020',  match_case=False)

        # why?
        # Scale the data from the MNE internal unit V to µV
        # ?
        samples = df.T#*1e-6
        
        raw = mne.io.RawArray(samples, info)
        raw.drop_channels(['Time', 'BlockNumber'])
        
        #high pass filter to remove slow drifts, 70 Hz low pass
        raw.filter(1., 70, None, fir_design='firwin')
        
        #remove power line interferance
        raw.notch_filter(50, n_jobs=-1)
        

        
        # Visual inspection of bad channels
        # TODO
        
        raw.set_eeg_reference('average', projection=True)

     
        #raw.set_eeg_reference(ref_channels=['Pz'])
        
        #raw.plot(scalings='20e+4')
        # raw.plot( scalings='20e-4', n_channels = 7, lowpass=bands.alpha[0], highpass=bands.alpha[1])
        # raw.plot_psd()
        
        # Create equal length epochs of 4 seconds
        epochs = mne.make_fixed_length_epochs(raw.copy(), preload=False, duration = 4)
    
  
        evoked = epochs.average()
        #evoked.plot_joint(picks='eeg')
        #evoked.plot_topomap(times=[0., 10., 20., 30., 90.], ch_type='eeg')
        
            
        # Autoreject based on rejection threshold
        # TODO
        # epochs.interpolate_bads()
        
        #reject = dict(eeg=400e-6)# unit: uV (EEG channels) dont forget the sample conversion to uV
        reject = get_rejection_threshold(epochs)
        reject['eeg'] = reject['eeg']
        print("The rejection dictionary is %s " %reject)
        # Remove bad epochs
        # TODO
        #epochs.drop_bad(reject=reject)  
        epochs.plot_drop_log()
        raw.plot()
        epochs.plot()
        


        # independent component analysis (ICA)
        # TODO
        ica = mne.preprocessing.ICA(method="fastica", n_components=7, random_state=97, max_iter='auto')
        ica.fit(epochs, reject=reject)
        #ica.fit(epochs)
        ica.plot_sources(epochs)
        ica.plot_components()
        #ica.plot_properties(epochs, picks=ica.exclude)


        ica.plot_overlay(epochs, exclude=[0, 2, 3, 4], picks='eeg')
        #ica.plot_overlay(raw, exclude=[2], picks='eeg')
        #ica.plot_overlay(raw, exclude=[3], picks='eeg')
        #ica.plot_overlay(raw, exclude=[4], picks='eeg')
        
        ica.exclude = [0, 2, 3 , 4] 
        # orig_raw = raw.copy()s
        # raw.load_data()
        epochs.load_data()
        ica.apply(epochs)   
        #ica.plot_sources(epochs)
  
   
        # set eeg reference
        
        #plot alpha and theta 
        if(False):
            # filter out alpha and theta
            # region
            raw_alpha = epochs.copy().filter(l_freq=bands.alpha[0], h_freq=bands.alpha[1], n_jobs=-1)
            # filter out Theta
            raw_theta = epochs.copy().filter(l_freq=bands.theta[0], h_freq=bands.theta[1], n_jobs=-1)
            
            fig = plt.figure( figsize=(7, 3))
            subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[3., 1.])
            axs0 = subfigs[0].subplots(2, 1)
            subfigs[0].set_facecolor('0.9')

            raw_alpha.plot_psd(ax = axs0[0],show=False, n_jobs=1)
            raw_theta.plot_psd(ax = axs0[1],show=False, n_jobs=1)
            
            axs1 = subfigs[1].subplots(2, 1)
            raw_theta.plot_psd_topo(axes = axs1[0],show=False, n_jobs=1)
            raw_theta.plot_psd_topo(axes = axs1[1],show=False, n_jobs=1)
            
            fig.set_constrained_layout(True)
            fig.suptitle("PID " + str(pid) + " block " + str(x))
            if(save_plots):
                filepath = "../Plots/PID_" + str(pid) + "-Block_" + str(x)  + "-raw_psd_topo.png"
                plt.savefig(filepath)

      
        ### Compute the power spectral density (PSD)
        
        group1 = epochs.copy().pick_channels(['F3', 'F4'])
        group2 = epochs.copy().pick_channels(['F3', 'F4', 'C3', 'C4'])
        group3 = epochs.copy().pick_channels(['P3', 'Pz', 'P4'])
        
        raw_groups = [group1, group2, group3, epochs.copy()]
        
        for grp_nr, raw_group in enumerate(raw_groups):
            raw = raw_group 

            picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                    stim=False)
                    
            method = ['Multitaper', 'Welch']
    
            for m in range(len(method)):

                if(method[m]) == 'Multitaper':
                    psds, freqs = psd_multitaper(epochs, low_bias=False,
                                proj=False, picks=picks,
                                n_jobs=-1, adaptive=False, normalization='length')
                    # Normalize the PSDs ?
                    psds /= np.sum(psds, axis=-1, keepdims=True) 
                    #convert to DB
                    #psds = 10 * np.log10(psds) * (-1) # erm lul wut
                elif(method[m]) == 'Welch':
                    psds, freqs = psd_welch(epochs,
                                proj=False, picks=picks,
                                n_jobs=2, n_overlap=150, n_fft=300)
                    # Normalize the PSDs ?
                    psds /= np.sum(psds, axis=-1, keepdims=True) 
                    #convert to DB
                    #psds = 10 * np.log10(psds) * (-1) # erm lul wut


                # # Normalize the PSDs ?
                # psds /= np.sum(psds, axis=-1, keepdims=True) 
                # #convert to DB
                # psds = 10 * np.log10(psds) * (-1) # erm lul wut

                #Mean of all channels
                psds_mean = psds.mean(0)
            
                freq_res = freqs[1] - freqs[0]
                
                # Find intersecting values in frequency vector
                idx_alpha = np.logical_and(freqs >= bands.alpha[0], freqs <= bands.alpha[1])
                idx_theta = np.logical_and(freqs >= bands.theta[0], freqs <= bands.theta[1])      
            
                # absolute power
                bp_alpha = simpson(psds_mean[idx_alpha], dx=freq_res)
                bp_theta = simpson(psds_mean[idx_theta], dx=freq_res) 
                bp_total = simpson(psds_mean, dx=freq_res)
                
                # relative power
                bp_alpha_rel =  bp_alpha / bp_total # alpha relative power
                bp_theta_rel = bp_theta / bp_total # theta relative power

                alpha_theta_total = bp_alpha / bp_theta
                alpha_theta_rel = bp_alpha_rel / bp_theta_rel       
                
                #peak power at freq
                peak_alpha = freqs[np.argmax(psds_mean[idx_alpha])]
                peak_theta = freqs[np.argmax(psds_mean[idx_theta])]

                # Extract the power values from the detected peaks
                # Plot the topographies across different frequency bands
                
                if(plot_plots):

                    fig, axes = plt.subplots(2, 2, figsize=(7, 3))
                    for ind, (label, band_def) in enumerate(bands):

                        # Get the power values across channels for the current band
                        f, psds1 = trim_spectrum(freqs, psds,  band_def)
                        
                        # Create a topomap for the current oscillation bandca
                        mne.viz.plot_topomap(psds1[:, 1], raw.info, cmap=cm.viridis,
                                            axes=axes[0, ind], show=False, ch_type='grad')
                        idx = np.logical_and(freqs >= band_def[0], freqs <=  band_def[1])
                        axes[0,ind].set_title(method[m] + " PSD topo " + label + ' power ' + str(channel_groups[grp_nr]), {'fontsize' : 7})

                        psds_std = (psds_mean[idx]).std(0)
                        peak = freqs[np.argmax(psds_mean[idx])]
                        axes[1,ind].plot(freqs[idx], psds_mean[idx], color='k')
                        axes[1,ind].fill_between(freqs[idx], psds_mean[idx] - psds_std, psds_mean[idx] + psds_std,
                                        color='k', alpha=.5)
                        axes[1,ind].set_title(method[m] + " PSD " + label + ' power', {'fontsize' : 7})
                    
                    fig.suptitle("PID " + str(pid) + " block " + str(x) + " " + str(channel_groups[grp_nr]))
                    fig.set_constrained_layout(True)
                    
                    if(save_plots):
                        filepath = "../Plots/PID_" + str(pid) + "-Block_" + str(x) + "-Group_" + str(grp_nr) + ".png"
                        plt.savefig(filepath)
            
                pws_lst.append([pid, x, bp_alpha, bp_theta, alpha_theta_total, grp_nr, method[m]])
                   
        if(draw_plots):

            plt.show()



# %%

f, axes = plt.subplots(4, 3, figsize=(15,6), constrained_layout=True)
dfPowers  = pd.DataFrame(pws_lst, columns =['PID', 'BlockNumber', 'AlphaPow', 'DeltaPow', 'AlphaTheta', 'Group', 'Method'])
ys = ['AlphaPow', 'DeltaPow', 'AlphaTheta']

for y in range(4):
    for i, ax1 in enumerate(axes[1]):
        sns.boxplot(x = "BlockNumber", y = ys[i], data = dfPowers.loc[(dfPowers['Group'] == y) & (dfPowers['Method'] == 'Multitaper')], ax=axes[y,i],showfliers=False)
        #sns.stripplot(x="BlockNumber", y = ys[i], data=dfPowers.loc[(dfPowers['Group'] == y) & (dfPowers['Method'] == 'Multitaper')], marker="o", alpha=0.3, color="black", ax=axes[y,i])
        axes[y,i].set_title( str(ys[i]) + " Group " + str(channel_groups[y]), fontsize=10)
        axes[y,i].set_ylabel('Power', fontsize=7)
        axes[y,i].set_xlabel('Block', fontsize=7)
f.suptitle("Multitaper Distribution")

f, axes = plt.subplots(4, 3, figsize=(15,6), constrained_layout=True)
for y in range(4):
    for i, ax1 in enumerate(axes[1]):
        sns.boxplot(x = "BlockNumber", y = ys[i], data = dfPowers.loc[(dfPowers['Group'] == y) & (dfPowers['Method'] == 'Welch')], ax=axes[y,i],showfliers=False)
        axes[y,i].set_title( str(ys[i]) + " Group " +  str(channel_groups[y]), fontsize=10)
        axes[y,i].set_ylabel('Power', fontsize=7)
        axes[y,i].set_xlabel('Block', fontsize=7)
f.suptitle("Welch Distribution")

plt.show()



