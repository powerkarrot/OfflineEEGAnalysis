#%%

import matplotlib.pyplot as plt
import mne
import philistine
from  Settings import *
from utils import *
import cupy as cp

mne.set_log_level(False)
mne.cuda.init_cuda(verbose=True)

mne.utils.set_config('MNE_USE_CUDA', 'true')  


print(mne.cuda.get_cuda_memory(kind='total'))

plt.rcParams.update({'figure.max_open_warning': 0})

raw_id1_eyesopen_novr = make_raw("ID101-EEG").crop(tmin = 4.0, tmax = 100.)
raw_id1_eyesclosed_novr = make_raw("ID103-EEG").crop(tmin = 4.0, tmax = 100.)
raw_id1_eyesopen_vr = make_raw("ID104-EEG").crop(tmin = 4.0, tmax = 100.)
raw_id1_eyesclosed_vr = make_raw("ID105-EEG").crop(tmin = 4.0, tmax = 100.)
raw_id1_eyesclosed2_vr = make_raw("ID107-EEG").crop(tmin = 4.0, tmax = 100.)
raw_id2_eyesopen_novr = make_raw("ID201-EEG").crop(tmin = 4.0, tmax = 100.)
raw_id2_eyesclosed_novr = make_raw("ID202-EEG").crop(tmin = 4.0, tmax = 100.)
raw_id2_eyesopen_vr = make_raw("ID203-EEG").crop(tmin = 4.0, tmax = 100.)
raw_id2_eyesclosed_vr = make_raw("ID204-EEG").crop(tmin = 4.0, tmax = 100.)

raws = [
        [[raw_id1_eyesopen_novr,raw_id1_eyesclosed_novr],[raw_id1_eyesopen_vr,raw_id1_eyesclosed_vr]],
        [[raw_id2_eyesopen_novr,raw_id2_eyesclosed_novr],[raw_id2_eyesopen_vr,raw_id2_eyesclosed_vr]]
       ]

# raw_id2_eyesopen_novr.plot_psd()
# raw_id2_eyesopen_novr.compute_psd(method='welch', fmin=4, fmax = 15, picks=[31]).plot()

# raws = [
#         [[raw_id2_eyesopen_novr,raw_id2_eyesclosed_novr],[raw_id2_eyesopen_vr,raw_id2_eyesclosed_vr]]
#        ]
# raw_id1_eyesopen_novr.pl



matplotlib.use('Agg') # supress plots
for i, subject in enumerate(raws):
    test = test_channels(subject[0][0],subject[0][1], channels)
    test2 = test_channels(subject[1][0],subject[1][1], channels)
    print("NO VR Subject", i+1, "problematic channels:" , test)
    print("WITH VR Subject", i+1, "problematic channels:" , test2)

    
for g, grp in enumerate(alpha_ch_groups):    
    for i, subject in enumerate(raws):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))        
        ax1.set_title("ID " + str(i+1) + " no VR ch_group " + str(g) )
        ax2.set_title("ID " + str(i+1) + " VR ch_group " + str(g)) 
        picks =  select_channels_picks(subject[0][0], grp)   
        novr_alpha = philistine.mne.attenuation_iaf([subject[0][0],subject[0][1]], picks=picks, savgol='diff', resolution=.1, ax=ax1)
        vr_alpha = philistine.mne.attenuation_iaf([subject[1][0],subject[1][1]], picks=picks, savgol='diff', resolution=.1,ax=ax2)
        print("ID", i+1 , " no VR ", "ch_group ", g, novr_alpha.AlphaBand[0] )
        print("ID", i+1 , " VR: ", "ch_group ", g, vr_alpha)
        fig.set_constrained_layout(True)
        
picks =  select_channels_picks(raw_id1_eyesclosed_vr, alpha_ch_groups[0])         
id1_vr_alpha_bothclosed = philistine.mne.attenuation_iaf([raw_id1_eyesclosed_vr,raw_id1_eyesclosed2_vr], resolution=.1,savgol='diff', picks=picks)
print("ID1 VR both closed: ", id1_vr_alpha_bothclosed)

picks =  select_channels_picks(raw_id1_eyesclosed_novr, alpha_ch_groups[0])         
v2_id2_vr_alpha = philistine.mne.savgol_iaf(raw_id1_eyesclosed_novr, picks=picks, resolution=.1)
print("V2 ID1 no VR eyes closed: ", v2_id2_vr_alpha)

# %%
