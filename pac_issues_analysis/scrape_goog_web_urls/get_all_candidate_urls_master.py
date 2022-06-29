# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 22:13:03 2021

@author: Shiyi Shen
"""

# imports
import os
import sys
from os import chdir, getcwd
from pathlib import Path

wd = getcwd()
chdir(wd)

# set directory to where this file is located
folder_loc = os.path.dirname(os.path.realpath(__file__))
os.chdir(Path(folder_loc))

# import other functions
# sys.path.insert(0, scripts_folder_rel_path)
from resources.functions.candidates_txt_to_csv import candidate_clean
from resources.functions.google_scraper_functions import goog_search
from resources.functions.google_scraper_functions import clean_camp_sites


#######################################################################

# set filepaths

# all the candidates
cand_file_rel_path = "resources/candidates/weball/weball22.txt"
# candidates descriptions
cand_descrip_rel_path = "resources/candidates/txt_description.csv"

# output.txt file name
outname = "candidates.csv"


# no need to change these filepaths
# functions folder
scripts_folder_rel_path = "resources/functions/"
# state abbreviations dataset
state_abbrev_rel_path = "resources/states/state_abbrev.csv"

#######################################################################
# # CANDIDATE FEC CSV IMPORT
# =============================================================================
# get candidate list
# cand_file = from FEC      cand_descrip = column names    state abbrev = state names
df = candidate_clean(cand_file_rel_path, cand_descrip_rel_path, state_abbrev_rel_path)



# # GET GOOGLE RESULTS
# =============================================================================
# create a column that we'll be searching with
df["search_query"] = (
    df["CAND_NAME"]
    + " "
    + df["CAND_PTY_AFFILIATION_FULL"]
    + " "
    + df["STATE_FULL"]
    + " "
    + df["ELECT_TYPE"]
)

# create campaign website specific columns
df["campaign_results_list"] = df["search_query"] + " " + "2021"


# search for campaign website results
for i, row in df.iterrows():
    row["campaign_results_list"] = goog_search(row['campaign_results_list'], 8, 7)

# additional cleaning for campaign site list
df["campaign_results_list"] = df.apply(clean_camp_sites, axis=1)
# CLEAN UP
del df["search_query"]
del df["twitter_search"]
del df["campaign_site_search"]

df.to_csv(outname, encoding="utf-8")
