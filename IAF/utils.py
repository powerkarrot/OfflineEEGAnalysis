
from itertools import chain, repeat
import os
from Settings import *
import pandas as pd
import mne
import matplotlib.pyplot as plt
    
def get_user_input(valid_response, prompt, err_prompt):
    prompts = chain([prompt], repeat(err_prompt))
    replies = map(input, prompts)
    lowercased_replies = map(str.lower, replies)
    stripped_replies = map(str.strip, lowercased_replies)
    return next(filter(valid_response.__contains__, stripped_replies))


def get_num_files(dir_path):
    return len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])

def make_raw(dir):

    dfEEG = pd.read_csv(f"{data_path}{dir}.csv")
    dfEEG.drop(["TimeLsl", "Time"], axis =1, inplace=True)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage('standard_1020',  match_case=False)
    samples = dfEEG.T
    raw = mne.io.RawArray(samples, info)
    
    return raw
