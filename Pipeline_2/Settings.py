from fooof.bands import Bands

ch_names = ['Time', 'FCz','Pz','P3','P4','PO7','PO8','Oz', 'BlockNumber']
ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',  'misc']

NUM_BLOCKS = 7

#channel_groups =[['F3', 'F4'],['F3', 'F4', 'C3', 'C4'],['P3', 'Pz', 'P4'],['F3','C3','P3','P4','C4','F4','Pz']]

channels = ['FCz','Pz','P3','P4','PO7','PO8','Oz']

alpha_ch_groups = [['Pz','P3','P4','PO7','PO8'],['FCz','Oz','PO8'], ['FCz','Pz','P3','P4','PO7','PO8','Oz'] ]
theta_ch_groups = [['FCz','Pz'], ['FCz','Pz'], ['FCz','Pz','P3','P4','PO7','PO8','Oz']]

assert(len(alpha_ch_groups) == len(theta_ch_groups))

channel_groups = [[alpha_ch_groups[0], theta_ch_groups[0], [[alpha_ch_groups[0], theta_ch_groups[0]]]], # fix this third thing
                  [alpha_ch_groups[1], theta_ch_groups[1], [[alpha_ch_groups[0], theta_ch_groups[0]]]],
                  [alpha_ch_groups[2], theta_ch_groups[2], [[alpha_ch_groups[0], theta_ch_groups[0]]]],
                 ] # always include all channels at end? 
print(len(channel_groups))

bands = Bands({'theta': [4, 8], 'alpha': [8, 12]})   
 
epochs_tstep = 4.

methods = ['multitaper', 'welch']

sfreq = 300



