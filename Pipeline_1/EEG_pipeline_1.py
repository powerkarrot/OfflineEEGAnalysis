# %%
import os
import pickle
from itertools import chain, repeat
from pathlib import Path

import autoreject
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from autoreject import get_rejection_threshold
from fooof.bands import Bands
from scipy.integrate import simpson

# %%
mne.set_log_level(False)
mne.utils.set_config('MNE_USE_CUDA', 'true')  
plt.rcParams.update({'figure.max_open_warning': 0})


# %%
def get_user_input(valid_response, prompt, err_prompt):
    prompts = chain([prompt], repeat(err_prompt))
    replies = map(input, prompts)
    lowercased_replies = map(str.lower, replies)
    stripped_replies = map(str.strip, lowercased_replies)
    return next(filter(valid_response.__contains__, stripped_replies))

# %%

NUM_BLOCKS = 7

channel_groups =[['F3', 'F4'],['F3', 'F4', 'C3', 'C4'],['P3', 'Pz', 'P4'],['F3','C3','P3','P4','C4','F4','Pz']]
ch_names = ['Time', 'F3','C3','P3','P4','C4','F4','Pz', 'BlockNumber']
ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',  'misc']

bands = Bands({'theta': [4, 8], 'alpha': [8, 12]})   
 
epochs_tstep = 4.


#%%
#Script config
plot_plots = False       
save_plots = False
draw_plots = False
pick_ic_auto = False

TEST = False

# %%
# bad channels
# TODO fill for all participants :')
# Format: bads[pid][block]
# example : bads[pid=1] = [['F2', 'F3'], [], ['C4'], [], [], [],  []]

bads = [[[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []],
        [[], [], [], [], [], [], []]
]

icas = []

# %%

lstPIds = []
path = "../Data/"
for filename in os.listdir(path):
    if filename.endswith(".csv"): 
        lstPIds.append(int(filename.split("-")[0].replace("ID", "")))
    else:
        continue
lstPIds = list(set(lstPIds))
print(lstPIds)
print(str(len(lstPIds)) + " subjects")

#for TESTING
if TEST:
    lstPIds = [1, 2]
    lstPIds = list(set(lstPIds))
    NUM_BLOCKS = 2

# %%

dir_path = r'./fifs'
Path('./fifs').mkdir(parents=True, exist_ok=True)
Path('./pickles').mkdir(parents=True, exist_ok=True)
Path('./Plots/ICA').mkdir(parents=True, exist_ok=True)
Path('./ica/fifs').mkdir(parents=True, exist_ok=True)

arr_epochs = []

if len(os.listdir('./fifs')) != NUM_BLOCKS * len(lstPIds):
                           
    for pid in tqdm.tqdm(lstPIds):
        
        # if (pid != 1):
        #    continue
        # if (pid > 2):
        #         break
        if len(os.listdir('./fifs/')) != NUM_BLOCKS * len(lstPIds):
        #if True:

        
            dfState = pd.read_csv(f"{path}ID{pid}-state.csv")
            dfState = pd.read_csv(f"{path}ID{pid}-state.csv")
                
            dfEEG = pd.read_csv(f"{path}ID{pid}-EEG.csv")
            dfEEG = dfEEG.rename(columns={"Value0": "F3", "Value1": "C3", "Value2": "P3", "Value3": "P4", "Value4": "C4", "Value5": "F4", "Value6": "Pz"})
            dfEEG.drop("TimeLsl", axis =1, inplace=True)

            dstate = pd.read_csv(f"{path}ID{pid}-state.csv")
            
            dfAll = pd.merge(dfEEG, dstate, on =["Time"], how="outer")
            dfAll = dfAll.sort_values(by="Time") # inplace?
            
            dfAll = dfAll.drop(columns=["Value7","AdaptationStatus", "NBackN", "State"] )
            dfAll.fillna(method='ffill', inplace=True)
            dfAll = dfAll.drop(dfAll[dfAll.BlockNumber < 0].index)
            dfAll = dfAll.dropna()

            for x in range(1, NUM_BLOCKS+1):  
                
                # if(x > 1):
                #     break
                
                # Prepare data 
                # see if fif already present, else filter / clean data and save it
                        
                data = dfAll.loc[dfAll['BlockNumber'] == x]
                df = pd.DataFrame(data)
                # data.plot(x="Time", y=["F3", "C3","P3","P3","C4","F4","Pz"])

                sfreq=300
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                info.set_montage('standard_1020',  match_case=False)

                samples = df.T
                
                raw = mne.io.RawArray(samples, info)
                raw.drop_channels(['Time', 'BlockNumber'])
                
                # remove power line interferance
                raw.notch_filter(50., n_jobs=-1)
                
                # high pass filter to remove slow drifts, 70 Hz low pass
                raw.filter(1., 70, None, fir_design='firwin')
                
                # set EEG reference
                raw.set_eeg_reference('average', projection=True)
                #raw.set_eeg_reference(ref_channels=['Pz'])
                

                # Visual inspection of bad channels
                # TODO, empty for now. With new setup, check for bad channels only once for all blocks.
                raw.info['bads'] =  bads[pid-1][x-1]
                if raw.info['bads']:
                    raw.interpolate_bads()
                
                #raw.plot(scalings='20e+4')
                # raw.plot( scalings='20e-4', n_channels = 7, lowpass=bands.alpha[0], highpass=bands.alpha[1])
                # raw.plot_psd()
                
                # Create equal length epochs of 4 seconds
                epochs = mne.make_fixed_length_epochs(raw.copy(), preload=True, duration = epochs_tstep)
                
                #evoked.plot_joint(picks='eeg')
                #evoked.plot_topomap(times=[0., 10., 20., 30., 90.], ch_type='eeg')
                
                # Global autoreject based on rejection threshold
                reject = get_rejection_threshold(epochs, ch_types = 'eeg', verbose=False)      
                #print("The rejection dictionary is %s " %reject)
                epochs.drop_bad(reject=reject)
                #epochs.plot_drop_log()
                #epochs.average().plot()                 
                
                #arr_epochs.append(epochs)
                epochs.save('./fifs/' + str(pid) + '-' + str(x) + '-epo.fif', overwrite = True)


# %%
exclude_ic = [] # TODO sure its here?
pick_ic_as_template = True
action = None

for pid in tqdm.tqdm(lstPIds):

    if action == 'no':
        pick_ic_as_template = False
    else:
        action = get_user_input(valid_response={'no', 'yes'},
                    prompt="Select ICs for ICE corrmap? - yes | no", 
                    err_prompt="Type  \"yes\" or \"no\": \n")
        
        if action == 'no':
            pick_ic_as_template = False
                
    for x in range(1, NUM_BLOCKS+1):  
        
        epochs = mne.read_epochs('./fifs/' + str(pid) + '-' + str(x) + '-epo.fif', preload=True)
        
        # should probably delete contents first but hey
        if len(os.listdir('./ica/fifs/')) != NUM_BLOCKS * len(lstPIds):
            # independent component analysis (ICA)
            
            ica = mne.preprocessing.ICA(method="fastica",  random_state=97)
            
            #TODO decide later. if yes, dont forget to redo it after ica apply
            local_autoreject = False
            if local_autoreject:
                ar = autoreject.AutoReject(n_jobs=-1,  verbose=True, random_state=11)
                epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True) 
                reject_log.plot('horizontal')
                evoked_bad = epochs[reject_log.bad_epochs].average()
                plt.figure()
                plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, 'r', zorder=-1)
                epochs_clean.average().plot(axes=plt.gca())
                #epochs = epochs_clean
                ica.fit(epochs[~reject_log.bad_epochs], tstep = epochs_tstep)

            ica.fit(epochs)
            ica.save('./ica/fifs/' + str(pid) + '-' + str(x) + '-ica.fif', overwrite = True)
            
        else:
            ica = mne.preprocessing.read_ica('./ica/fifs/' + str(pid) + '-' + str(x) + '-ica.fif')
         
        # Optionally autoreject muscle and eog artifacts
        if (pick_ic_auto):
            #start fresh, else find_bads_muscle fails
            ica.exclude = []
            #print(str(pid) + ' block ' + str(x))
            
            ica_z_thresh = 1.96
            eog_indices, eog_scores = ica.find_bads_eog(epochs, 
                                                        ch_name=['F3', 'F4'], 
                                                        threshold=ica_z_thresh)
            #print(f'Automatically found eye artifact ICA components: {eog_indices}')

            # ica.plot_scores(eog_scores)  
            # ica.plot_overlay(epochs, exclude=eog_indices, picks='eeg', title = str(pid) + '-' + str(x) )
            muscle_idx_auto = []
            
            #TODO make true again, just not with  this data...
            if(False):
                muscle_idx_auto, scores = ica.find_bads_muscle(epochs)
                #ica.plot_scores(scores, exclude=muscle_idx_auto)
                
                #print(f'Automatically found muscle artifact ICA components: {muscle_idx_auto}')
                #ica.plot_overlay(epochs.average(), exclude=muscle_idx_auto, picks='eeg', title = str(pid) + '-' + str(x))
            
            for item in muscle_idx_auto + eog_indices :
                if item not in ica.exclude:
                    ica.exclude.append(item)
                    
            #print("excludes are" , ica.exclude)
            #ica.plot_overlay(epochs.average(), exclude=ica.exclude, picks='eeg', title = str(pid) + '-' + str(x) )

        # Pick templates
        if(pick_ic_as_template):
            done = False
            while not done:
                
                ica.plot_properties(epochs, dB= True, log_scale= True, psd_args={'fmax':70})
                ica.plot_sources(epochs, block = True, title = str(pid) + '-' + str(x), stop = 360. )
                ics_old = ica.exclude

                ica.plot_overlay(epochs.average(), exclude=exclude_ic, picks='eeg', stop = 360.)

                while True:
                    accept = get_user_input(valid_response={'no', 'yes'},
                                            prompt="Accept? - yes | no",
                                            err_prompt = "yes | no")
                    try:
                        if accept == 'yes':
                            exclude_ic = ica.exclude
                            #ica.exclude = [] # avoid excluding it twice
                            ready_to_write = True                            
                            if(ready_to_write):
                                ica.save('./ica/'+ str(pid) + '-' + str (x) + '_template-ica.fif', overwrite = True)
                                ica.save('./ica/fifs/' + str(pid) + '-' + str(x) + '-ica.fif', overwrite = True)
                                done = True
                                quit = get_user_input(valid_response={'no', 'yes'},
                                            prompt="Quit? - yes | no",
                                            err_prompt = "yes | no")                   
                                if quit == 'yes':
                                    pick_ic_as_template = False
                                    done = True
                                    break
                            break
                        else:
                            ica.exclude = ics_old # doesn't to anything
                            quit = get_user_input(valid_response={'no', 'yes'},
                                            prompt="Quit? - yes | no",
                                            err_prompt = "yes | no")                       
                            if quit == 'yes':
                                pick_ic_as_template = False
                                done = True
                                break     
                            else:
                                break              
                    except Exception as e:
                        print(e)
        
        # TODO remove save ica from template picker  
        # if there are excludes in ica, template doesnt work :DDDD      
        ica.save('./ica/fifs/' + str(pid) + '-' + str(x) + '-ica.fif', overwrite = True)       
           
        #TODO put this somewhere else
        clean_ica_excludes = False
        if(clean_ica_excludes):
            print("removing excludes")
            ica.exclude = []
            ica.save('./ica/fifs/' + str(pid) + '-' + str(x) + '-ica.fif', overwrite = True)   
              
        #TODO maybe do a size check before appending                                  
        # save the ICAs for the corrmap 
        icas.append(ica)
        arr_epochs.append(epochs)
        # epochs.save("./ica/pipeline_1/raw/"+str(pid)+"_"+str(x)+".fif")
    
 
 # %%
ica_templates = []
ica_excludes = []

#count = 0
dir_path = r'./ica/'
for path in os.scandir(dir_path):
    if path.is_file():
        #count += 1
        #print(path.name)
        ica_template = mne.preprocessing.read_ica(dir_path  + path.name)
        if(ica_template.exclude != []):           
            ica_templates.append(ica_template)
        else:
            os.remove(dir_path + path.name)
 
clean_epochs = np.empty((len(lstPIds), NUM_BLOCKS), dtype=object) # remove

for n, ic_templ in enumerate(ica_templates):
    icas.insert(0,ic_templ) #set template
    for x, excl in enumerate(ica_templates[n].exclude):
        #threshold=0.9
        mne.preprocessing.corrmap(icas, [0,excl], label='exclude', threshold=0.9, plot=False)
    icas.pop(0) # remove template.
    
p = 0
b = 0
for i, n in enumerate(icas):
    b += 1
    p = p + 1 if i % 7 == 0 else p
    if p == 4: p = 5
    if p == 8:  p = 9
    if p == 10: p = 11
    b = 1 if  b == 8 else b
    
    #print(p , " block ", b)

    if 'exclude' in n.labels_:
        #n.plot_overlay(arr_epochs[i].average(), n.labels_['exclude'], picks='eeg', title=("p "+ str(p) +" block " +str(b)) )
        
        #add autodetected artifacts to exclude
        if(pick_ic_auto):
            for item in  n.labels_['exclude'] :
                if item not in n.exclude:
                    n.exclude.append(item)
        else:
            n.exclude = n.labels_['exclude']
    else:
        print("No templates selected \n")
    
    #n.exclude = []
    #n.exclude = [0,1,2,3,4,5]
    #print("Final ICAs to exclude are" ,n.exclude)
    #n.plot_overlay(arr_epochs[i].average(), n.exclude, picks='eeg',  title=("p "+ str(p) +" block " +str(b)))
    n.apply(arr_epochs[i]) # TODO at least i hope so, double check indices. 

clean_epochs = np.reshape(arr_epochs, (len(lstPIds),NUM_BLOCKS))
#clean_epochs = np.reshape(arr_epochs, (2,2)) # for testing only

# TODO save preprocessed epochs. (somewhere else)      
#raw.save("./ica/pipeline_1/raw/"+str(p)+"_"+str(x)+".fif")
    
#%%
pws_lst = list()

for n, pid in enumerate(tqdm.tqdm(lstPIds)):
    # if (pid > 2):
    #         break
    for x in range(1, NUM_BLOCKS+1):  
        
        #epochs = mne.io.read_raw_fif("./ica/pipeline_1/raw/"+str(pid)+"_"+str(x)+".fif")
        epochs = clean_epochs[n][x-1] 

        #epochs = arr_epochs[((x-1)*len(lstPIds)) + pid]
        
        # Average all epochs
        evoked = epochs.average()     
           
        #Plot alpha and theta PSD
        if(plot_plots):
    
            fig = plt.figure( figsize=(7, 3))
            subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[3., 1.])
            axs0 = subfigs[0].subplots(2, 1)
            subfigs[0].set_facecolor('0.9')

            epochs.compute_psd(method='multitaper', fmin=4, fmax = 8).plot(dB=False, axes = axs0[1], show = False)
            epochs.compute_psd(method='multitaper', fmin=8, fmax = 12).plot(dB = False, axes = axs0[0], show = False) #  .plot_topomap({'alpha': (8,12)},  normalize=True, axes=axes[1], show=False)
        
            axs1 = subfigs[1].subplots(2, 1)
            evoked.compute_psd(method='multitaper', fmin=4, fmax = 8).plot_topo(dB = False, axes = axs1[0], show = False)
            evoked.compute_psd(method='multitaper', fmin=8, fmax = 12).plot_topo(dB = False, axes = axs1[1], show = False)
            
            fig.set_constrained_layout(True)
            fig.suptitle("PID " + str(pid) + " block " + str(x))
            if(save_plots):
                filepath = "./Plots/PID_" + str(pid) + "-Block_" + str(x)  + "-raw_psd_topo.png"
                plt.savefig(filepath)

      
        ### Compute the power spectral density (PSD)
        group1 = epochs.copy().pick_channels(['F3', 'F4'])
        group2 = epochs.copy().pick_channels(['F3', 'F4', 'C3', 'C4'])
        group3 = epochs.copy().pick_channels(['P3', 'Pz', 'P4'])
        
        raw_groups = [group1, group2, group3, epochs.copy()]
        
        for grp_nr, epochs in enumerate(raw_groups):
            evoked = epochs.average() 

            picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False,
                    stim=False)
                    
            methods = ['multitaper', 'welch']
    
            for m, method in enumerate(methods):
                
                spectrum = epochs.compute_psd(method = method, n_jobs=-1)
                # average across epochs first
                mean_spectrum = spectrum.average() # TODO whaaaaaaaaaaaaaaaaaaaaa?
                psds, freqs = mean_spectrum.get_data(return_freqs=True)
        
                # Normalize the PSDs ?
                # psds /= np.sum(psds, axis=-1, keepdims=True) 
                
                # #convert to DB
                psdsDB = 10 * np.log10(psds)

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

                        # Create a topomap for the current oscillation band
                        epochs.compute_psd(method=method).plot_topomap({label: band_def}, ch_type='eeg',cmap='viridis', show_names=True, normalize=True, axes=axes[0, ind], show=False)
                        
                        axes[0,ind].set_title(method + " PSD topo " + label + ' power ' + str(channel_groups[grp_nr]), {'fontsize' : 7})
                        
                        idx = np.logical_and(freqs >= band_def[0], freqs <=  band_def[1])
                        psds_std = (psds_mean[idx]).std(0)
                        peak = freqs[np.argmax(psds_mean[idx])]
                        axes[1,ind].plot(freqs[idx], psds_mean[idx], color='k')
                        axes[1,ind].fill_between(freqs[idx], psds_mean[idx] - psds_std, psds_mean[idx] + psds_std,
                                        color='k', alpha=.5)
                        axes[1,ind].set_title(method + " PSD " + label + ' power', {'fontsize' : 7})
                    
                    fig.suptitle("PID " + str(pid) + " block " + str(x) + " " + str(channel_groups[grp_nr]))
                    fig.set_constrained_layout(True)
                    
                    if(save_plots):
                        filepath = "./Plots/PID_" + str(pid) + "-Block_" + str(x) + "-Group_" + str(grp_nr) + ".png"
                        plt.savefig(filepath)
            
                pws_lst.append([pid, x, bp_alpha, bp_theta, alpha_theta_total, grp_nr, method])
                   
        if(draw_plots):

            plt.show()



# %%

f, axes = plt.subplots(4, 3, figsize=(15,6), constrained_layout=True)
dfPowers  = pd.DataFrame(pws_lst, columns =['PID', 'BlockNumber', 'AlphaPow', 'DeltaPow', 'AlphaTheta', 'Group', 'Method'])
ys = ['AlphaPow', 'DeltaPow', 'AlphaTheta']

for y in range(4):
    for i, ax1 in enumerate(axes[1]):
        sns.boxplot(x = "BlockNumber", y = ys[i], data = dfPowers.loc[(dfPowers['Group'] == y) & (dfPowers['Method'] == 'multitaper')], ax=axes[y,i],showfliers=False)
        #sns.stripplot(x="BlockNumber", y = ys[i], data=dfPowers.loc[(dfPowers['Group'] == y) & (dfPowers['Method'] == 'Multitaper')], marker="o", alpha=0.3, color="black", ax=axes[y,i])
        axes[y,i].set_title( str(ys[i]) + " Group " + str(channel_groups[y]), fontsize=10)
        axes[y,i].set_ylabel('Power', fontsize=7)
        axes[y,i].set_xlabel('Block', fontsize=7)
f.suptitle("Multitaper Distribution")

f, axes = plt.subplots(4, 3, figsize=(15,6), constrained_layout=True)
for y in range(4):
    for i, ax1 in enumerate(axes[1]):
        sns.boxplot(x = "BlockNumber", y = ys[i], data = dfPowers.loc[(dfPowers['Group'] == y) & (dfPowers['Method'] == 'welch')], ax=axes[y,i],showfliers=False)
        axes[y,i].set_title( str(ys[i]) + " Group " +  str(channel_groups[y]), fontsize=10)
        axes[y,i].set_ylabel('Power', fontsize=7)
        axes[y,i].set_xlabel('Block', fontsize=7)
f.suptitle("Welch Distribution")

plt.show()

# %%

dfPowers.to_pickle('./pickles/dfPowers.pickle')


# %%
