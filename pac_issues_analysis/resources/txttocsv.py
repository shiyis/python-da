import pandas as pd
import os
import numpy as np
pd.set_option("display.max_colwidth", -1)
from text_cleaning.cleaning_functions import *
pp = Preprocessing()
dfs = []
for top, dir, files in os.walk('/Users/shiyishen/PycharmProjects/pac_issues/data/'):
  for filename in files:
    file_path = os.path.join(top,filename)

    read_file = pd.read_csv(rf'{file_path}', header = None,sep='\n',error_bad_lines = False)
    read_file.columns = ['text']
    name = str(' '.join(filename.split('_')[1:]).split('.')[0]).upper()
    
    
    
    
    read_file = pp.create_explode_df(read_file,'text')
    read_file = read_file[read_file['text'] != 'None']
    read_file.drop_duplicates(subset=['text'])
    read_file['name'] = name
    columns_titles = ["name","text","count"]
    read_file=read_file.reindex(columns=columns_titles)
    dfs.append(read_file)

df = pd.concat(dfs, axis=0)
df = df.sort_values('text', ascending=False)
df = df.drop_duplicates(subset=['text'])
df = df.sort_values('name',ascending=True)
df.reset_index(drop=True, inplace=True)
df.to_csv ('/Users/shiyishen/PycharmProjects/pac_issues/inputs/pac_fund_issues.csv')



    
    