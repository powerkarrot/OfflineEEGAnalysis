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
            ica = mne.preprocessing.read_ica('./ica/fifs/' + str(pid) + '-' + str(block) + '-ica.fif') 
            ica.plot_sources(epochs, block = True)
            exclude_ic = ica.exclude
            #ica.exclude = [] # avoid excluding it twice
            
            ica.plot_overlay(epochs.average(), exclude=exclude_ic, picks='eeg', stop = 360.)
            
            while True:
                accept = input("Accept? - yes | esc")
                try:
                    if accept == 'yes':
                        exclude_ic = ica.exclude
                        #ica.exclude = [] # avoid excluding it twice
                        ica.save('./ica/fifs/' + str(pid) + '-' + str(block) + '-ica.fif', overwrite = True)
                        count = 0
                        dir_path = r'./ica/'
                        for path in os.scandir(dir_path):
                            if path.is_file():
                                count += 1
                        count /= 2
                        with open('./ica/ica_template-' + str(int(count)) + '.pickle', 'wb') as f:
                            pickle.dump(ica, f)
                        with open('./ica/exclude-'+ str(int(count)) + '.pickle', 'wb') as f:
                            pickle.dump(exclude_ic, f)
                        done = True
                        break
                    else:
                        break                   
                except:
                    print("oof")
                    
        except Exception as e:
            print("Invalid input.", e)
            



# %%
