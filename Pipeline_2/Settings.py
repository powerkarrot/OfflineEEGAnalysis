from fooof.bands import Bands

#Script config
plot_plots = False       
save_plots = False
draw_plots = False
pick_ic_auto = False
pick_ic_as_template = False

TEST = False


bands = Bands({'theta': [4, 8], 'alpha': [8, 12]})   
epochs_tstep = 4.
methods = ['multitaper', 'welch']
sfreq = 300
NUM_BLOCKS = 7


ch_names = ['Time', 'FCz','Pz','P3','P4','PO7','PO8','Oz', 'BlockNumber']
ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',  'misc']

channels = ['FCz','Pz','P3','P4','PO7','PO8','Oz']
alpha_ch_groups = [['Pz','P3','P4','PO7','PO8'],['FCz','Oz','PO8'],  ['FCz','Pz'],['FCz','Pz','P3','P4','PO7','PO8','Oz'] ]
theta_ch_groups = [['FCz','Pz'], ['FCz','Pz'], ['FCz','Pz'], ['FCz','Pz','P3','P4','PO7','PO8','Oz']]

assert(len(alpha_ch_groups) == len(theta_ch_groups))

channel_groups = [[alpha_ch_groups[0], theta_ch_groups[0], [[alpha_ch_groups[0], theta_ch_groups[0]]]], #TODO choose channels for ratio
                  [alpha_ch_groups[1], theta_ch_groups[1], [[alpha_ch_groups[1], theta_ch_groups[1]]]],
                  [alpha_ch_groups[2], theta_ch_groups[2], [[alpha_ch_groups[2], theta_ch_groups[2]]]],
                  [alpha_ch_groups[3], theta_ch_groups[3], [[alpha_ch_groups[3], theta_ch_groups[3]]]],

                 ] # always include all channels at end? 


# bad channels
# TODO fill for all participants and blocks :')
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







