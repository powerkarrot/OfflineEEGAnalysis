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
            epochs = mne.read_epochs('./fifs/' + str(pid) + '-' + str(block) + '-epo.fif')
            epochs.load_data()
            ica = mne.preprocessing.read_ica('./ica/fifs/' + str(pid) + '-' + str(block) + '-ica.fif') 
            ica.plot_sources(epochs, block = True)
            exclude_ic = ica.exclude
            #ica.exclude = [] # avoid excluding it twice
            
            ica.plot_overlay(epochs.average(), exclude=exclude_ic, picks='eeg', stop = 360.)
            
            while True:
                #TODO User input validation? Who cares!
                accept = input("Accept? - yes | esc")
                try:
                    if accept == 'yes':
                        exclude_ic = ica.exclude
                        #ica.exclude = [] # avoid excluding it twice
                        ica.save('./ica/fifs/' + str(pid) + '-' + str(block) + '-ica.fif', overwrite = True)
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
