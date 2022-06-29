import nltk
import string as s
import pandas as pd
import re
import sys
from nltk.corpus import stopwords
from typing import Sequence,List,Tuple
from tqdm import tqdm
tqdm.pandas()
# import contractions
# ============================================================
# PART 1: BASIC PREPROCESSING
# tokenization
# remove punctuation
# remove stopwords
# remove numbrs and non-ascii
# remove empty string
# remove words w length 1
# expanding contractions
#
# PART2: ADVANCED PROCESSING
# replace contra words w periods
# remove phrases like 'google translate'
# replace punctuation w period
# remove all single char
# remove long/short text
#
# PART3: TOKENIZE AND EXPLODE TEXT
# tokenize text with nltk sent_tok
# explode text into dataframe
# return dataframe
# ============================================================


class Preprocessing:

  def __init__(self):
    self.contra_words = ['however ', 'but ', 'although', 'nevertheless', 'yet']                                   # special processing w contra words                 
                                                                                                                  # remove translaed terms
    self.punctuation = ''.join([i for i in s.punctuation if i not in ',.;?!~'])





  def text_clean(self,text) -> str :
    text = [''.join([i for i in word if i not in self.punctuation and ord(i) < 128]).rstrip() # remove numbers and non-ascii
            for word in text.lower().rstrip().split()
            if word not in self.punctuation]                            # remove punctuation
    text = ' '.join([word for word in text if word !=''])         # get rid of empty string
    text.replace('|'.join(self.contra_words), '. ')               # replace contra words with period
    if len(text) > 20 and ('www' or '/' or 'http') not in text:
        return text




  def create_explode_df(self, df, text) -> List[Sequence]:
    df[text] = df[text].apply(str).apply(self.text_clean)
    df[text] = df[text].apply(str).apply(nltk.sent_tokenize)
    df[text] = df[text].apply(lambda x: " ".join(x))

    df = df[pd.notnull(df[text])]                                   # get rid of null values and text either too long or too short
    df = df[df[text].str.split().str.len() > 40]
    df = df[pd.notnull(df[text])]
    df['count'] = df[text].str.split().str.len()
    return df

