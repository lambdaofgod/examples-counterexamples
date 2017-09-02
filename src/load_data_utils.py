"""
 this script assumes its functions are called from some of project's subdirectories
"""


import os


def get_env_vars():
    PROJECT_DIR = os.split(os.getcwd())[0]
    DATA_DIR = os.join(PROJECT_DIR, 'data')
    return PROJECT_DIR, DATA_DIR
