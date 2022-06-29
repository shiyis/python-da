# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:54:05 2020

@author: NewUsername
"""


import pandas as pd
import os
import sys
from os import chdir, getcwd
from tqdm import tqdm
wd = getcwd()  # lets you navigate using chdir within jupyter/spyder
chdir(wd)

tqdm.pandas()

def candidate_clean(cand_file_rel_path, cand_descrip_rel_path, state_rel_path):
    # import cand file
    cand_df = pd.read_csv(cand_file_rel_path, sep="|", header=None)
    # import description so we get column names
    cand_descrip = pd.read_csv(cand_descrip_rel_path, encoding='utf-8')
    
    col_names = cand_descrip['Column name'].tolist()
    
    
    # set col names
    cand_df.columns = col_names
    # get only relevant columns
    cand_df = cand_df[['CAND_ID', 'CAND_NAME', 'CAND_PTY_AFFILIATION',
                       'CAND_OFFICE_ST', 'CAND_OFFICE_DISTRICT', 'CAND_ICI']]
    
    # create senate house pres column



    cand_df['ELECT_TYPE'] = cand_df['CAND_ID'].apply(type_election)

    # new column for full state name
    state_abbrev_df = pd.read_csv(state_rel_path, encoding='utf-8')
    cand_df = pd.merge(cand_df, state_abbrev_df, left_on='CAND_OFFICE_ST',
                       right_on='STATE_ABBREV', how='left')
    del cand_df['STATE_ABBREV']
    cand_df['STATE_FULL'] = cand_df['STATE_FULL'].fillna('')
    # create full name of party



    
    cand_df['CAND_PTY_AFFILIATION_FULL'] = cand_df['CAND_PTY_AFFILIATION'].apply(
        party_full)
    
    cand_df['CAND_ICI'] = cand_df['CAND_ICI'].map({'C':'Challenger','I':'Incumbent','O':'Open seat'})
    print(cand_df['CAND_ICI'])

    return cand_df


def party_full(text):
    if text == 'IND':
        return 'Independent'
    elif text == 'DEM':
        return 'Democrat'
    elif text == 'REP':
        return 'Republican'
    elif text == 'GRE':
        return 'Green'
    elif text == 'LIB':
        return 'Libertarian'
    else:
        return 'Other'
    
    
def type_election(text):
    if text[0] == 'S':
        return 'Senate'
    elif text[0] == 'H':
        return 'House'
    elif text[0] == 'P':
        return 'President'
    else:
        return ''


def incumbency(text):
    if text[0] == "C":
        return 'Challenger'
    elif text[0] == 'O':
        return 'Open seat'
    else:
        return 'Incumbent'