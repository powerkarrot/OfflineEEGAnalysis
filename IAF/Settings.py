from fooof.bands import Bands

sfreq = 300
data_path = './LogData/'

ch_names = ['Fp1','Fz','F3','F7','F9','FC5','FC1','C3','T7','CP5','CP1',
            'Pz','P3','P7','P9','O1','Oz','O2','P10','P8','P4','CP2','CP6',
            'T8','C4','Cz','FC2','FC6','F10','F8','F4','Fp2']
ch_types = ['eeg'] * 32

channels = ['Fp1','Fz','F3','F7','F9','FC5','FC1','C3','T7','CP5','CP1',
            'Pz','P3','P7','P9','O1','Oz','O2','P10','P8','P4','CP2','CP6',
            'T8','C4','Cz','FC2','FC6','F10','F8','F4','Fp2']

alpha_ch_groups = [channels] # otherwise out of bounds error/cant automatically detect upper end of alpha band
# alpha_ch_groups = [['Pz','P3','P4','O1','Oz','O2'],
#                    channels
#                    ]


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







