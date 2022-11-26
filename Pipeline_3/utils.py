
from itertools import chain, repeat
import os
    
def get_user_input(valid_response, prompt, err_prompt):
    prompts = chain([prompt], repeat(err_prompt))
    replies = map(input, prompts)
    lowercased_replies = map(str.lower, replies)
    stripped_replies = map(str.strip, lowercased_replies)
    return next(filter(valid_response.__contains__, stripped_replies))


def get_num_files(dir_path):
    return len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])