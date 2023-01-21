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
    suma = 0
    for i in range(len(w2) - len(w1.split()) + 1):
        w2_help = w2[i:i + len(w1.split())]
        w2_help = ' '.join(w2_help)
        if difflib.SequenceMatcher(None, w2_help, w1).ratio() > treshhold:
            suma+=1
    return suma


slownik = []

polemo_category = "medicine_text"
preprocessed = "all"
function = "all"

if (polemo_category == "all_text"):
    polemo_categories = ["hotels_text", "medicine_text", "products_text", "reviews_text"]
elif (polemo_category == "all_sentence"):
    polemo_categories = ["hotels_sentence", "medicine_sentence", "products_sentence", "reviews_sentence"]
else:
    polemo_categories = [polemo_category]

if (function == "all"):
    functions = ["sbert", "tfidf", "textrank"]
else:
    functions = [function]

if (preprocessed == "all"):
    preprocessing_categories = ["simple", "lemmatization", "spelling"]  # TODO add 'spelling' when done
else:
    preprocessing_categories = [preprocessed]

nlp = spacy.load("pl_core_news_lg")

threshold_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

for category in polemo_categories:
    for preprocessing_category in preprocessing_categories:
        for function in functions:
            for threshold1 in threshold_list:
                for threshold2 in threshold_list:
                    if function == "sbert":
                        models = ["_distiluse-base-multilingual-cased-v1", "_Multilingual-MiniLM-L12-H384", "_all-distilroberta-v1"]
                    if function == "textrank":
                        models = ["_textrank", "_positionrank", "_topicrank", "_biasedtextrank"]
                    if function == "tfidf":
                        models = [""]
                    for model in models:


                        filename = f"out/out6_validation/{category}_{preprocessing_category}_{function}{model}_{threshold1}_{threshold2}_no_duplicates.txt"
                        slownik_b=[]
                        slownik_g=[]
                        with open(filename, "w", encoding="utf-8") as file:
                            for b in import_slownik(f"out/{function}/{category}_{preprocessing_category}_{function}{model}_4000_1.txt"):
                                slownik_b.append(b)
                            for g in import_slownik(f"out/{function}/{category}_{preprocessing_category}_{function}{model}_4000_2.txt"):
                                slownik_g.append(g)

                            print(slownik_b)
                            print(slownik_g)
                            slownik_b,slownik_g = remove_shared_words_from_dictionary(slownik_b,slownik_g)
                            slownik_b = slownik_b[:100]
                            slownik_g = slownik_g[:100]
                            print(slownik_b)
                            print(slownik_g)

                            df = pd.read_csv(f'data/{category}_validation_{preprocessing_category}_preprocessed.csv', sep=";", header=0)

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

                            poprawne_neutralne_dobre = 0
                            poprawne_neutralne_zle = 0
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
                                sentyment_pozytywne = 0
                                sentyment_negatywnie = 0
                                zdanie = row['text']
                                # print(zdanie)
                                sentyment_prawdziwy = row['target']

                                score = 0
                                sumab = 0

                                threshold = len(row["text"].split()) * threshold2

                                for words in slownik_b:
                                    sentyment_negatywnie += diff(words, row["text"].split(), threshold1)
                                    if sentyment_negatywnie>threshold:
                                        break
                                sumag = 0
                                for words in slownik_g:
                                    sentyment_pozytywne += diff(words, row["text"].split(), threshold1)
                                    if sentyment_pozytywne>threshold:
                                        break


                                print(sentyment_pozytywne, sentyment_negatywnie, threshold)

                                if sentyment_pozytywne >= threshold and (sentyment_prawdziwy == 2 or sentyment_prawdziwy == 3):
                                    poprawne_dobre += 1
                                elif sentyment_pozytywne < threshold and (sentyment_prawdziwy == 0 or sentyment_prawdziwy == 1):
                                    poprawne_neutralne_dobre += 1
                                elif sentyment_pozytywne >= threshold and (sentyment_prawdziwy == 0 or sentyment_prawdziwy == 1):
                                    neutralne_jako_dobre += 1
                                elif sentyment_pozytywne < threshold and (sentyment_prawdziwy == 2 or sentyment_prawdziwy == 3):
                                    dobre_jako_neutralne += 1

                                if sentyment_negatywnie >= threshold and (sentyment_prawdziwy == 1 or sentyment_prawdziwy == 3):
                                    poprawne_zle += 1
                                elif sentyment_negatywnie < threshold and (sentyment_prawdziwy == 0 or sentyment_prawdziwy == 2):
                                    poprawne_neutralne_zle += 1
                                elif sentyment_negatywnie >= threshold and (sentyment_prawdziwy == 0 or sentyment_prawdziwy == 2):
                                    neutralne_jako_zle += 1
                                elif sentyment_negatywnie < threshold and (sentyment_prawdziwy == 1 or sentyment_prawdziwy == 3):
                                    zle_jako_neutralne += 1


                                lacznie += 1

                            poprawne = poprawne_neutralne_dobre + poprawne_neutralne_zle + poprawne_dobre + poprawne_zle

                            file.write(f"{poprawne} / {lacznie*2}\n")
                            file.write(f"{poprawne / (lacznie * 2) * 100} % poprawnie przypisanych \n")
                            # print(slowa, sentyment, dobry, zly)

                            file.write(f"{poprawne_dobre + poprawne_neutralne_dobre} / {lacznie}\n")
                            file.write(f"{(poprawne_dobre + poprawne_neutralne_dobre) / lacznie * 100}, % poprawnie przypisanych pozytywnych \n")

                            file.write(f"{poprawne_zle + poprawne_neutralne_zle} / {lacznie}\n")
                            file.write(f"{(poprawne_zle + poprawne_neutralne_zle) / lacznie * 100}, % poprawnie przypisanych zlych \n \n")

                            file.write(f"{poprawne_dobre} {neutralne_jako_dobre} {poprawne_neutralne_dobre} {dobre_jako_neutralne}\n\n")
                            file.write(tabulate([['Pozytywne', poprawne_dobre, neutralne_jako_dobre], ['Niepozytywne', dobre_jako_neutralne, poprawne_neutralne_dobre]], headers=['oryginalne\przypisane', 'Pozytywne', 'Niepozytywne']))
                            file.write("\n\n")
                            file.write(tabulate([['Negatywne', poprawne_zle, neutralne_jako_zle], ['Nienegatywne', zle_jako_neutralne, poprawne_neutralne_zle]], headers=['oryginalne\przypisane', 'Negatywne','Nienegatywne']))
