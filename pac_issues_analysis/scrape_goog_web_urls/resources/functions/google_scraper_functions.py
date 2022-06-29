# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:31:22 2020

@author: NewUsername
"""

from googlesearch import search


def goog_search(term, num_search, pausing):
    # print the string we're searching
    print("term, ", term)
    
    # empty array to populate with results
    fill_array = []

    search_list = search(term,num_results=10)
    for j in search_list:  
        fill_array.append(j)
        print(j)



    # remove everything after com, net, etc and check for duplicates
    com_net = ['com/', 'net/', 'org/']
    com_net = '|'.join(com_net)

    # remove anything after .com .net etc in a url
    fill_array = [x.lower().split(com_net)[0] for x in fill_array]
    # remove dups
    fill_array = list(dict.fromkeys(fill_array))

    # check for ballotpedia, and other random websites that we don't want
    # !! in the future make facebook scraper
    extraneous_websites = ['ballotpedia', 'upi', 'govtrack', 'herald', 'wikipedia',
                           'ydr', 'facebook', 'twitter', 'adn', '.gov']
    fill_array = [x for x in fill_array if not any(
        ext in x for ext in extraneous_websites)]
    # print('campaign site', fill_array)
    return fill_array



def clean_camp_sites(row):
    # clean up the sites and url and stuff
    try:
        names_lower = row['CAND_NAME'].lower().split()
        # remove middle initials
        names_lower = [x for x in names_lower if len(x) > 1]
        # remove any special characters
        names_lower = [x.replace(',', '') for x in names_lower]
        href_list = row['campaign_results_list']
        href_list = ['/'.join(x.lower().split('/')[:4]) for x in href_list]

        fill_array = []
        for x in href_list:

            if any(n in x for n in names_lower):
                fill_array.append(x)

        return fill_array
    except:
        return ['']
