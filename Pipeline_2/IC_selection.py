# %%
import os
import mne
import pickle
import matplotlib.pyplot as plt

# %%

action = None
done = False

while not done:
        action = input("Type pid and blocknumber or esc to exit")
        pid = action.split()[0]
        block = action.split()[1]
        try:
            raw = mne.io.read_raw_fif('./fifs/' + str(pid) + '-' + str(block) + '_eeg.fif')
            raw.load_data()
            ica = mne.preprocessing.read_ica('./ica/fifs/' + str(pid) + '-' + str(block) + '-ica.fif') 
            ica.plot_sources(raw, block = True)
            exclude_ic = ica.exclude
            #ica.exclude = [] # avoid excluding it twice
            
            ica.plot_overlay(raw, exclude=exclude_ic, picks='eeg', stop = 360.)
            
            while True:
                #TODO User input validation? Who cares!
                accept = input("Accept? - yes | esc")
                try:
                    if accept == 'yes':
                        exclude_ic = ica.exclude
                        #TODO reconsider this below.
                        #ica.exclude = [] # avoid excluding it twice
                        ica.save('./ica/fifs/' + str(pid) + '-' + str(block) + '-ica.fif', overwrite = True) # ahh.
                        ica.save('./ica/'+ str(pid) + '-' + str (block) + '_template-ica.fif', overwrite = True)
                        done = True
                        break
                    else:
                        break                   
                except:
                    print("oof")
                    
        except Exception as e:
            print("Invalid input.", e)
            



# %%
