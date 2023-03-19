
from itertools import chain, repeat
from Settings import *
import mne
from itertools import compress
import numpy as np
from distutils.util import strtobool

    
def get_user_input(valid_response, prompt, err_prompt):
    prompts = chain([prompt], repeat(err_prompt))
    replies = map(input, prompts)
    lowercased_replies = map(str.lower, replies)
    stripped_replies = map(str.strip, lowercased_replies)
    return next(filter(valid_response.__contains__, stripped_replies))

def get_psd(instance, method, picks, n):
    spectrum = instance.compute_psd(method = method, n_jobs=n, picks=picks)
    if(method == 'multitaper'):
        # kwargs = {'adaptive':True}
        kwargs = {'low_bias': False, 'bandwidth': None}
        spectrum = instance.compute_psd(method = method, n_jobs=n, picks=picks, **{'low_bias': False})
    else:
        kwargs = {'n_fft':1024}
        try:
            spectrum = instance.compute_psd(method = method, n_jobs=n, picks=picks, **{'n_fft':1024})
        except:
            spectrum = instance.compute_psd(method = method, n_jobs=n, picks=picks, **{'n_fft':128})

    return spectrum.get_data(return_freqs=True)

def mask_channels(target_ch):
    mask = np.array(np.isin(channels, target_ch, invert=True), dtype = bool)
    excl = list(compress(channels, mask))
    return excl

def psdsDB(psd):
     return 10 * np.log10(psd)