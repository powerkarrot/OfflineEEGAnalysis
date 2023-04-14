from fooof.bands import Bands

#Script config
plot_plots = False       
save_plots = False
draw_plots = False
pick_ic_auto = False
pick_ic_as_template = False



bands = Bands({'theta': [4, 8], 'alpha': [8, 12]})   
epochs_tstep = 4.
methods = ['multitaper', 'welch']
sfreq = 500
NUM_BLOCKS = 5 #change to 9, then remove the resting blocks
START_BLOCK = 1

TEST = False

path = "../Data/"
path = "../PilotData/15%/"
path = "../PilotData/30%/"


ch_names = ['Time', 'Fp1','Fz','F3','F7','F9','FC5','FC1','C3','T7','CP5','CP1',
            'Pz','P3','P7','P9','O1','Oz','O2','P10','P8','P4','CP2','CP6',
            'T8','C4','Cz','FC2','FC6','F10','F8','F4','Fp2','AF7','AF3','AFz'
            ,'F1','F5','FT7','FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','Iz'
            ,'POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','F2','AF4','AF8', 'BlockNumber']

#ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',  'misc']
ch_types = ['misc'] + ['eeg'] * 64 + ['misc']

channels = ['Fp1','Fz','F3','F7','F9','FC5','FC1','C3','T7','CP5','CP1',
            'Pz','P3','P7','P9','O1','Oz','O2','P10','P8','P4','CP2','CP6',
            'T8','C4','Cz','FC2','FC6','F10','F8','F4','Fp2','AF7','AF3','AFz'
            ,'F1','F5','FT7','FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','Iz'
            ,'POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','F2','AF4','AF8']

alpha_ch_groups = [
                   ['P3','Pz','PO3','POz','PO4','O1','O2'],
                    channels
                   ]
theta_ch_groups = [['Fp1','Fp2','AFz','AF3','AF4','F1','F2','F3','Fz','F4','FC1','FC2'], 
                    channels
                  ]

assert(len(alpha_ch_groups) == len(theta_ch_groups))

channel_groups = [[alpha_ch_groups[0], theta_ch_groups[0], [[alpha_ch_groups[0], theta_ch_groups[0]]]], #TODO choose channels for ratio
                  [alpha_ch_groups[1], theta_ch_groups[1], [[alpha_ch_groups[1], theta_ch_groups[1]]]],
                 ] # always include all channels at end? 


# bad channels
# TODO fill for all participants :') oh and expand to 32 channels later :DDDDDDDDDDDDDDDDD
# Format: bads[pid][block]
# example : bads[pid=1] = [['F2', 'F3'], [], ['C4'], [], [], [],  []]
bads = [[[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []],
        [[], [], [], [], [], [],  []]
]







