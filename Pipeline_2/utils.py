
from itertools import chain, repeat
from Settings import *
import mne
from itertools import compress
import numpy as np
    
def get_user_input(valid_response, prompt, err_prompt):
    prompts = chain([prompt], repeat(err_prompt))
    replies = map(input, prompts)
    lowercased_replies = map(str.lower, replies)
    stripped_replies = map(str.strip, lowercased_replies)
    return next(filter(valid_response.__contains__, stripped_replies))

def get_psd(instance, method, picks, n):
    spectrum = instance.compute_psd(method = method, n_jobs=n, picks=picks)
    return spectrum.get_data(return_freqs=True)

def mask_channels(target_ch):
    mask = np.array(np.isin(channels, target_ch, invert=True), dtype = bool)
    excl = list(compress(channels, mask))
    return excl

def psdsDB(psd):
     return 10 * np.log10(psd)