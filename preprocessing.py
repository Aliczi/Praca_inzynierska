#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from datasets import load_dataset


polemo_category = "hotels_text"
polemo_official = load_dataset("data/polemo2-official/", polemo_category) # only oppinions about hotels
df_polemo_official = pd.DataFrame(polemo_official["train"])
print(df_polemo_official.head)

import sys
sys.path.insert(0, './spellcorrectorpl/python')
from spellcorrectorpl.python.KnownWordsProvider import KnownWordsProviderUsingRAM, KnownWordsProviderUsingBigFile, KnownWordsProviderUsingMultipleFiles
from spellcorrectorpl.python.BigramsProvider import BigramsProvider
from spellcorrectorpl.python.SpellCorrector import SpellCorrector
import re

UNIGRAMS_FILEPATH = '1grams_fixed'
UNIGRAMS_FILES_DIR = '1grams_splitted'
BIGRAMS_FILEPATH = "2grams_splitted"

words_provider = KnownWordsProviderUsingMultipleFiles()
bigrams_provider = BigramsProvider()
words_provider.initialize(UNIGRAMS_FILES_DIR)
bigrams_provider.initialize(BIGRAMS_FILEPATH)

corrector = SpellCorrector(words_provider, bigrams_provider)

def correct_opinion(opinion: str, corrector: SpellCorrector):        
    opinion = re.sub(r'\d', "<liczba>", opinion) #change numbers to #L
    opinion = corrector.sentence_correction(opinion, print_words=False)
    return opinion

from string import punctuation
import unidecode
import csv
import re


def preprocess(df: pd.DataFrame, path):
    translate_table = dict((ord(char), None) for char in punctuation)
    oppinions = df.copy()
    with open(path, 'a', encoding='UTF8', newline='') as f:
        
        oppinions["text"] = oppinions["text"].str.replace('"',"") # remove quotation
        oppinions["text"] = oppinions["text"].str.translate(translate_table) # remove punctuation
        oppinions["text"] = oppinions["text"].str.casefold() # to lower
        oppinions["text"] = oppinions["text"].str.replace('[^łśćżźąęńóa-zA-Z\s\n\.]', '', regex=True)
        oppinions["text"] = oppinions["text"].str.replace(r'\s+', ' ', regex=True) # remove quotation
        for i in range(933, len(oppinions["text"])):

            oppinions.at[i,"text"] = correct_opinion(oppinions.at[i,"text"], corrector=corrector)
            oppinions.at[i,"text"] = unidecode.unidecode(oppinions.at[i,"text"]) # remove polish characters
            
            if(i==0):
                print(oppinions.loc[[i]])
                #oppinions.loc[[i]].to_csv(path, sep=';')
                #oppinions.loc[[i]].to_csv(path, sep=';', header=False,mode='a')
            else:
                print(oppinions.loc[[i]])
                #oppinions.loc[[i]].to_csv(path, sep=';', header=False,mode='a')
    return oppinions

print(df_polemo_official)
#test_df = pd.DataFrame([["jesli wybierasz sie na zamek z dziecmi  co wiecej oczy trzeba miec doslownie z kazdej strony", 3]], columns=["text", "target"])
#oppinions = preprocess(test_df,"data/opinions_hotels_preprocessed.csv")
oppinions = preprocess(df_polemo_official,"data/opinions_hotels_preprocessed.csv")
#oppinions = preprocess(df_polemo_official,"data/hotels_text_train_simple_preprocessed.csv")





