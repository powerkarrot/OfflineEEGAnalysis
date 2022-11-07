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
