#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import spacy
import difflib
from tabulate import tabulate

def remove_shared_words_from_dictionary(dict1, dict2):
    for word1 in dict1:
        if word1 in dict2:
            dict1.remove(word1)
            dict2.remove(word1)
    return dict1, dict2


def import_slownik(nazwa_pliku):
    f = open(nazwa_pliku, "r")
    slownik_tab = []
    for line in f.readlines():
        # print([line.strip()])
        slownik_tab.append(line.strip())
    f.close()
    return slownik_tab


def similarity(w1, w2, treshhold):
    w1_nlp = nlp(w1)
    suma2 = 0
    for i in range(len(w2) - len(w1.split()) + 1):
        w2_help = w2[i:i + len(w1.split())]
        w2_help = ' '.join(w2_help)
        w2_help_nlp = nlp(w2_help)
        if w2_help_nlp.similarity(w1_nlp) > treshhold:
            return 1
    return 0

def diff(w1, w2, treshhold):
    suma2 = 0
    for i in range(len(w2) - len(w1.split()) + 1):
        w2_help = w2[i:i + len(w1.split())]
        w2_help = ' '.join(w2_help)
        if difflib.SequenceMatcher(None, w2_help, w1).ratio() > treshhold:
            return 1
    return 0


slownik = []

polemo_category = "all_text"
preprocessed = "all"
function = "all"

if (polemo_category == "all_text"):
    polemo_categories = ["hotels_text", "medicine_text", "products_text", "reviews_text"]
else:
    polemo_categories = [polemo_category]

if (function == "all"):
    functions = ["sbert", "tfidf", "textrank"]
else:
    functions = [function]

if (preprocessed == "all"):
    preprocessing_categories = ["simple", "lemmatization"]  # TODO add 'spelling' when done
else:
    preprocessing_categories = [preprocessed]

nlp = spacy.load("pl_core_news_lg")

for category in polemo_categories:
    for preprocessing_category in preprocessing_categories:
        for function in functions:
            filename = f"out/out3/{category}_{preprocessing_category}_{function}.txt"
            slownik_b=[]
            slownik_g=[]
            with open(filename, "w", encoding="utf-8") as file:
                for b in import_slownik(f"out/{function}/{category}_{preprocessing_category}_{function}_1.txt"):
                    slownik_b.append(b)
                for g in import_slownik(f"out/{function}/{category}_{preprocessing_category}_{function}_2.txt"):
                    slownik_g.append(g)

                print(slownik_b)
                print(slownik_g)
                slownik_b,slownik_g = remove_shared_words_from_dictionary(slownik_b,slownik_g)
                print(slownik_b)
                print(slownik_g)

                df = pd.read_csv(f'data/{category}_test_{preprocessing_category}_preprocessed.csv', sep=";", header=0)

                lacznie_neutralne = 0
                lacznie_dobre = 0
                lacznie_zle = 0

                for index, row in df.iterrows():
                    if row['target'] == 0 or row['target'] == 3:
                        lacznie_neutralne += 1
                    elif row['target'] == 2:
                        lacznie_dobre += 1
                    elif row['target'] == 1:
                        lacznie_zle += 1

                print('neutralne: ', lacznie_neutralne)
                print('dobre: ', lacznie_dobre)
                print('zle: ', lacznie_zle)

                lacznie = 0

                poprawne_neutralne = 0
                poprawne_dobre = 0
                poprawne_zle = 0

                neutralne_jako_dobre = 0
                neutralne_jako_zle = 0

                dobre_jako_neutralne = 0
                dobre_jako_zle = 0

                zle_jako_neutralne = 0
                zle_jako_dobre = 0

                for index, row in df.iterrows():

                    dobry, zly = 0, 0
                    sentyment = 0
                    zdanie = row['text']
                    # print(zdanie)
                    sentyment_prawdziwy = row['target']

                    score = 0
                    sumab = 0
                    for words in slownik_b:
                        sumab += diff(words, row["text"].split(), 0.75)
                    sumag = 0
                    for words in slownik_g:
                        sumag += diff(words, row["text"].split(), 0.75)
                    sentyment = sumag - sumab

                    if sentyment == 0 and (
                            sentyment_prawdziwy == 0 or sentyment_prawdziwy == 3):
                        poprawne_neutralne += 1
                    elif sentyment > 0 and sentyment_prawdziwy == 2:
                        poprawne_dobre += 1
                    elif sentyment < 0 and sentyment_prawdziwy == 1:
                        poprawne_zle += 1

                    elif sentyment < 0 and (
                            sentyment_prawdziwy == 0 or sentyment_prawdziwy == 3):
                        neutralne_jako_zle += 1
                    elif sentyment > 0 and (
                            sentyment_prawdziwy == 0 or sentyment_prawdziwy == 3):
                        neutralne_jako_dobre += 1

                    elif sentyment == 0 and sentyment_prawdziwy == 2:
                        dobre_jako_neutralne += 1
                    elif sentyment < 0 and sentyment_prawdziwy == 2:
                        dobre_jako_zle += 1

                    elif sentyment == 0 and sentyment_prawdziwy == 1:
                        zle_jako_neutralne += 1
                    elif sentyment > 0 and sentyment_prawdziwy == 1:
                        zle_jako_dobre += 1


                    lacznie += 1

                poprawne = poprawne_neutralne + poprawne_dobre + poprawne_zle

                file.write(f"{poprawne / lacznie}\n")
                file.write(f"{poprawne / lacznie * 100}, % poprawnie przypisanych \n")
                # print(slowa, sentyment, dobry, zly)

                file.write(f"{poprawne_neutralne / lacznie_neutralne}\n")
                file.write(f"{poprawne_neutralne / lacznie_neutralne * 100}, % poprawnie przypisanych neutralnych \n")

                file.write(f"{poprawne_dobre / lacznie_dobre}\n")
                file.write(f"{poprawne_dobre / lacznie_dobre * 100}, % poprawnie przypisanych dobrych \n")

                file.write(f"{poprawne_zle / lacznie_zle}\n")
                file.write(f"{poprawne_zle / lacznie_zle * 100}, % poprawnie przypisanych zlych \n")

                file.write(tabulate([['Neutralne', poprawne_neutralne, neutralne_jako_dobre, neutralne_jako_zle],
                                ['Dobre', dobre_jako_neutralne, poprawne_dobre, dobre_jako_zle],
                                ['Zle', zle_jako_neutralne, zle_jako_dobre, poprawne_zle]],
                               headers=['oryginalne\przypisane', 'Neutalne', 'Dobre', 'Zle']))
