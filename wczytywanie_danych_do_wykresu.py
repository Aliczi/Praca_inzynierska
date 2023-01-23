import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

polemo_category = "all_text"
preprocessed = "simple"
function = "all"

if (polemo_category == "all_text"):
    polemo_categories = ["hotels_text", "products_text", "reviews_text"]
elif (polemo_category == "all_sentence"):
    polemo_categories = ["hotels_sentence", "medicine_sentence", "products_sentence", "reviews_sentence"]
else:
    polemo_categories = [polemo_category]

if (function == "all"):
    functions = ["sbert", "tfidf", "textrank"]
else:
    functions = [function]

if (preprocessed == "all"):
    preprocessing_categories = ["simple", "lemmatization", "spelling"]
else:
    preprocessing_categories = [preprocessed]

threshold_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
models = [""]



for threshold1 in threshold_list:
    with open(f"out/wykres/{threshold1}.csv", "w", encoding="utf-8") as file2:
        file2.write(f"t2, acc\n")
        for threshold2 in threshold_list:
            licznik = 0
            mianownik = 0
            for category in polemo_categories:
                for preprocessing_category in preprocessing_categories:
                    for function in functions:
                        if function == "sbert":
                            models = ["_distiluse-base-multilingual-cased-v1"]
                        if function == "textrank":
                            models = ["_textrank", "_positionrank", "_topicrank", "_biasedtextrank"]
                        if function == "tfidf":
                            models = [""]
                        for model in models:
                            filename_wszystkie = f"out/out6_validation/{category}_{preprocessing_category}_{function}{model}_0.1_0.1_no_duplicates.txt"
                            filename_zadne = f"out/out6_validation/{category}_{preprocessing_category}_{function}{model}_1_1_no_duplicates.txt"
                            with open(filename_wszystkie, "r", encoding="utf-8") as file_wszystkie:
                                list = file_wszystkie.readline().split(" ")
                                wszystkie = int(list[0]) / int(list[2])
                            with open(filename_zadne, "r", encoding="utf-8") as file_zadne:
                                list = file_zadne.readline().split(" ")
                                zadne = int(list[0]) / int(list[2])
                            filename = f"out/out6_validation/{category}_{preprocessing_category}_{function}{model}_{threshold1}_{threshold2}_no_duplicates.txt"
                            with open(filename, "r", encoding="utf-8") as file:
                                list = file.readline().split(" ")
                                list = [int(list[0]), int(list[2])]
                                if(int(list[0])/int(list[1])>max(wszystkie,zadne)):
                                    licznik += int(list[0])/int(list[1])-max(wszystkie,zadne)
            print(licznik)
            file2.write(f"{threshold2}, {licznik}\n")
