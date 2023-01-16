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
function = "all"

if (polemo_category == "all_text"):
    polemo_categories = ["hotels_text", "medicine_text", "products_text", "reviews_text"]
else:
    polemo_categories = [polemo_category]

if (function == "all"):
    functions = ["tfidf", "sbert", "textrank"]
else:
    functions = [function]

if (preprocessed == "all"):
    preprocessing_categories = ["simple", "lemmatization"]  # TODO add 'spelling' when done
else:
    preprocessing_categories = [preprocessed]


for category in polemo_categories:
    for preprocessing_category in preprocessing_categories:
        for function in functions:
            filename = f"out/out2/{category}_{preprocessing_category}_{function}.txt"
            slownik = []
            with open(filename, "w", encoding="utf-8") as file:
                for b in import_slownik(f"out/{function}/{category}_{preprocessing_category}_{function}_1.txt"):
                    slownik.append([len(b),b,-1])

                for g in import_slownik(f"out/{function}/{category}_{preprocessing_category}_{function}_2.txt"):
                    slownik.append([len(g),g,1])

                slownik = sorted(slownik, key=lambda x: x[0], reverse=True)

                lacznie_neutralne = 0
                lacznie_dobre = 0
                lacznie_zle = 0
                lacznie_nieprzypisane = 0

                df = pd.read_csv(f'data/{category}_test_{preprocessing_category}_preprocessed.csv', sep=";", header=0)

                for index, row in df.iterrows():
                    if row['target'] == 0:
                        lacznie_neutralne += 1
                    elif row['target'] == 3:
                        lacznie_nieprzypisane += 1
                    elif row['target'] == 2:
                        lacznie_dobre += 1
                    elif row['target'] == 1:
                        lacznie_zle += 1

                lacznie_negatywne = lacznie_dobre + lacznie_nieprzypisane
                lacznie_nienegatywne = lacznie_zle + lacznie_neutralne

                lacznie_negatywne = lacznie_zle + lacznie_nieprzypisane
                lacznie_nienegatywne = lacznie_dobre + lacznie_neutralne

                file.write(f'Wszystkich opinii: {lacznie_negatywne+lacznie_nienegatywne}\n\n')

                file.write(f'Pozytwyne: {lacznie_dobre}\n')
                file.write(f'Niepozytwyne: {lacznie_neutralne}\n\n')

                file.write(f'Negatywne: {lacznie_zle}\n')
                file.write(f'Nienegatywne: {lacznie_nienegatywne}\n\n')



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
                    #print(zdanie)
                    sentyment_prawdziwy = row['target']

                    for n in slownik:
                        if n[1] in zdanie:
                            if(n[2]==1):
                                sentyment_pozytywne += zdanie.count(n[1])
                            elif(n[2]==-1):
                                sentyment_negatywnie += zdanie.count(n[1])
                            #print(zdanie)
                            zdanie = zdanie.replace(n[1], "")
                            #print(zdanie)

                    if sentyment_pozytywne >= 1 and (sentyment_prawdziwy == 2 or sentyment_prawdziwy == 3):
                        poprawne_dobre += 1
                    elif sentyment_pozytywne == 0 and (sentyment_prawdziwy == 0 or sentyment_prawdziwy == 1):
                        poprawne_neutralne_dobre += 1
                    elif sentyment_pozytywne >= 1 and (sentyment_prawdziwy == 0 or sentyment_prawdziwy == 1):
                        neutralne_jako_dobre += 1
                    elif sentyment_pozytywne == 0 and (sentyment_prawdziwy == 2 or sentyment_prawdziwy == 3):
                        dobre_jako_neutralne += 1

                    if sentyment_negatywnie >= 1 and (sentyment_prawdziwy == 1 or sentyment_prawdziwy == 3):
                        poprawne_zle += 1
                    elif sentyment_negatywnie == 0 and (sentyment_prawdziwy == 0 or sentyment_prawdziwy == 2):
                        poprawne_neutralne_zle += 1
                    elif sentyment_negatywnie >= 1 and (sentyment_prawdziwy == 0 or sentyment_prawdziwy == 2):
                        neutralne_jako_zle += 1
                    elif sentyment_negatywnie == 0 and (sentyment_prawdziwy == 1 or sentyment_prawdziwy == 3):
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
                print(file.name)