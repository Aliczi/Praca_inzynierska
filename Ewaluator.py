#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from tabulate import tabulate

def import_slownik(nazwa_pliku):
    f = open(nazwa_pliku, "r")
    slownik_tab = []
    for line in f.readlines():
        # print([line.strip()])
        slownik_tab.append(line.strip())
    f.close()
    return slownik_tab

slownik = []

polemo_category = "all_text"
preprocessed = "all"
function = "textrank"

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

if (preprocessed != "none"):
    for category in polemo_categories:
        for preprocessing_category in preprocessing_categories:
            for function in functions:
                filename = f"out/out/{category}_{preprocessing_category}_{function}.txt"
                with open(filename, "w", encoding="utf-8") as file:
                    for b in import_slownik(f"out/{function}/{category}_{preprocessing_category}_{function}_1.txt"):
                        slownik.append([len(b),b,-1])

                    for g in import_slownik(f"out/{function}/{category}_{preprocessing_category}_{function}_2.txt"):
                        slownik.append([len(g),g,1])

                    slownik = sorted(slownik, key=lambda x: x[0], reverse=True)

                    lacznie_neutralne = 0
                    lacznie_dobre = 0
                    lacznie_zle = 0

                    df = pd.read_csv(f'data/{category}_test_{preprocessing_category}_preprocessed.csv', sep=";", header=0)

                    for index, row in df.iterrows():
                        if row['target'] == 0 or row['target'] == 3:
                            lacznie_neutralne += 1
                        elif row['target'] == 2:
                            lacznie_dobre += 1
                        elif row['target'] == 1:
                            lacznie_zle += 1

                    file.write(f'neutralne: {lacznie_neutralne}\n')
                    file.write(f'dobre: {lacznie_dobre}\n')
                    file.write(f'zle: {lacznie_zle}\n')

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
                        #print(zdanie)
                        sentyment_prawdziwy = row['target']

                        for n in slownik:
                            if n[1] in zdanie:
                                sentyment += zdanie.count(n[1])*n[2]
                                #print(zdanie)
                                zdanie = zdanie.replace(n[1], "")
                                #print(zdanie)

                        if sentyment == 0 and (sentyment_prawdziwy == 0 or sentyment_prawdziwy == 3):
                            poprawne_neutralne += 1
                        elif sentyment > 0 and sentyment_prawdziwy == 2:
                            poprawne_dobre += 1
                        elif sentyment < 0 and sentyment_prawdziwy == 1:
                            poprawne_zle += 1

                        elif sentyment < 0 and (sentyment_prawdziwy == 0 or sentyment_prawdziwy == 3):
                            neutralne_jako_zle += 1
                        elif sentyment > 0 and (sentyment_prawdziwy == 0 or sentyment_prawdziwy == 3):
                            neutralne_jako_dobre += 1

                        elif sentyment == 0 and sentyment_prawdziwy == 2:
                            dobre_jako_neutralne += 1
                        elif sentyment < 0 and sentyment_prawdziwy == 2:
                            dobre_jako_zle += 1

                        elif sentyment == 0 and sentyment_prawdziwy == 1:
                            zle_jako_neutralne += 1
                        elif sentyment > 0 and sentyment_prawdziwy == 1:
                            zle_jako_dobre += 1

                        else:
                            print(sentyment, sentyment_prawdziwy)

                        lacznie += 1

                    poprawne = poprawne_neutralne + poprawne_dobre + poprawne_zle

                    file.write(f"{poprawne} / {lacznie}\n")
                    file.write(f"{poprawne / lacznie * 100} % poprawnie przypisanych \n")
                    # print(slowa, sentyment, dobry, zly)

                    file.write(f"{poprawne_neutralne} / {lacznie_neutralne}\n")
                    file.write(f"{poprawne_neutralne / lacznie_neutralne * 100}, % poprawnie przypisanych neutralnych \n")

                    file.write(f"{poprawne_dobre} / {lacznie_dobre}\n")
                    file.write(f"{poprawne_dobre / lacznie_dobre * 100}, % poprawnie przypisanych dobrych \n")

                    file.write(f"{poprawne_zle} / {lacznie_zle}\n")
                    file.write(f"{poprawne_zle / lacznie_zle * 100}, % poprawnie przypisanych zlych \n \n")

                    file.write(tabulate([['Neutralne', poprawne_neutralne,neutralne_jako_dobre,neutralne_jako_zle], ['Dobre', dobre_jako_neutralne, poprawne_dobre, dobre_jako_zle], ['Zle',zle_jako_neutralne, zle_jako_dobre, poprawne_zle]], headers=['oryginalne\przypisane','Neutalne', 'Dobre', 'Zle']))
                    print(file.name)