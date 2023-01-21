#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from datasets import load_dataset

import sys


sys.path.insert(0, './spellcorrectorpl/python')
from spellcorrectorpl.python.KnownWordsProvider import KnownWordsProviderUsingRAM, KnownWordsProviderUsingBigFile, \
    KnownWordsProviderUsingMultipleFiles
from spellcorrectorpl.python.BigramsProvider import BigramsProvider
from spellcorrectorpl.python.SpellCorrector import SpellCorrector
import spacy
from spacy.lang.pl.examples import sentences


#!python -m spacy download pl_core_news_sm #TODO aby urchomić należy pobrać tę bazę
nlp = spacy.load("pl_core_news_sm")
lemmatizer = nlp.get_pipe("lemmatizer")

UNIGRAMS_FILEPATH = '1grams_fixed'
UNIGRAMS_FILES_DIR = '1grams_splitted'
BIGRAMS_FILEPATH = "2grams_splitted"

words_provider = KnownWordsProviderUsingMultipleFiles()
bigrams_provider = BigramsProvider()
words_provider.initialize(UNIGRAMS_FILES_DIR)
bigrams_provider.initialize(BIGRAMS_FILEPATH)

corrector = SpellCorrector(words_provider, bigrams_provider)


def correct_opinion(opinion: str, corrector: SpellCorrector):
    opinion = re.sub(r'\d', "<liczba>", opinion)  # change numbers to #L
    opinion = corrector.sentence_correction(opinion, print_words=False)
    return opinion

def lemmatization(opinion: str):
    opinion = nlp(opinion)
    lemmas = []
    for token in opinion:
        lemmas.append(token.lemma_)
    return ' '.join(lemmas)

from string import punctuation
import unidecode
import re


def preprocess(df: pd.DataFrame, path, option):
    translate_table = dict((ord(char), None) for char in punctuation)
    oppinions = df.copy()
    with open(path, 'a', encoding='UTF8', newline='') as f:

        oppinions["text"] = oppinions["text"].str.replace('"', "")  # remove quotation
        oppinions["text"] = oppinions["text"].str.translate(translate_table)  # remove punctuation
        oppinions["text"] = oppinions["text"].str.casefold()  # to lower
        oppinions["text"] = oppinions["text"].str.replace('[^łśćżźąęńóa-zA-Z\s\n\.]', '', regex=True)
        oppinions["text"] = oppinions["text"].str.replace(r'\s+', ' ', regex=True)  # remove quotation
        for i in range(len(oppinions["text"])):
            if option == 2:
                oppinions.at[i, "text"] = lemmatization(oppinions.at[i, "text"])
            if option == 3:
                oppinions.at[i, "text"] = correct_opinion(oppinions.at[i, "text"], corrector=corrector)
            oppinions.at[i, "text"] = unidecode.unidecode(oppinions.at[i, "text"])  # remove polish characters

            if (i == 0):
                print(oppinions.loc[[i]])
                oppinions.loc[[i]].to_csv(path, sep=';')
                # oppinions.loc[[i]].to_csv(path, sep=';', header=False,mode='a')
            else:
                print(oppinions.loc[[i]])
                oppinions.loc[[i]].to_csv(path, sep=';', header=False,mode='a')
    return oppinions


option = 3  # 1 simple, 2 lemmatization, 3 spelling corrector
files = 6  # 1 all, 2 hotels, 3 medicine, 4 products, 5 reviews
train_or_test = 1  # 1 all, 2 train, 3 test, 4 validate

if files == 1:
    categories = ["hotels_text", "medicine_text", "products_text", "reviews_text"]
elif files == 2:
    categories = ["hotels_text"]
elif files == 3:
    categories = ["medicine_text"]
elif files == 4:
    categories = ["products_text"]
elif files == 5:
    categories = ["reviews_text"]
else:
    categories = ["hotels_sentence", "medicine_sentence", "products_sentence", "reviews_sentence"]

if train_or_test == 1:
    data_sets = ["train", "test", "validation"]
elif train_or_test == 2:
    data_sets = ["train"]
elif train_or_test == 4:
    data_sets = ["validation"]
else:
    data_sets = ["test"]

for polemo_category in categories:
    polemo_official = load_dataset("data/polemo2-official/", polemo_category)
    for data_set in data_sets:
        df_polemo_official = pd.DataFrame(polemo_official[data_set])
        print(df_polemo_official)
        # test_df = pd.DataFrame([["jesli wybierasz sie na zamek z dziecmi  co wiecej oczy trzeba miec doslownie z kazdej strony", 3]], columns=["text", "target"])
        # oppinions = preprocess(test_df,"data/opinions_hotels_preprocessed.csv")
        if option == 1:
            oppinions = preprocess(df_polemo_official, "data/" + polemo_category + "_" + data_set + "_simple_preprocessed.csv", 1)
        elif option == 2:
            oppinions = preprocess(df_polemo_official, "data/" + polemo_category + "_" + data_set + "_lemmatization_preprocessed.csv", 2)
        elif option == 3:
            oppinions = preprocess(df_polemo_official, "data/" + polemo_category + "_" + data_set + "_spelling_preprocessed.csv", 3)
        else:
            oppinions = preprocess(df_polemo_official, "data/opinions_hotels_preprocessed.csv", 3)
        # oppinions = preprocess(df_polemo_official,"data/hotels_text_train_simple_preprocessed.csv")
