"""
 this script assumes its functions are called from some of project's subdirectories
"""

import os
from os.path import join, split 


def get_env_vars(go_up=False):
    if go_up:
        PROJECT_DIR = split(os.getcwd())[0]
    else:
        PROJECT_DIR = os.getcwd()
    DATA_DIR = join(PROJECT_DIR, 'data')
    return PROJECT_DIR, DATA_DIR
