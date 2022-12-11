import pandas as pd
from datasets import load_dataset
import os


def save_dicts_to_files(dicts: dict, prefix: str):
    """dictionaries with keywords should be in form {class0: [keyword0, keyword1, ...], ...}"""
    directory = "out"
    for sentiment in dicts:
        filename = f"{directory}/{prefix}_{sentiment}.txt"
        os.makedirs(directory, exist_ok=True)
        with open(filename, "w") as file:
            file.write("\n".join(dicts[sentiment]))
        print(f"Keywords for sentiment {sentiment} written to file {filename}")


def load_raw_data(data_path: str, data_category: str, traintest: str = "train"):
    polemo_official = load_dataset(data_path, data_category)
    df_polemo_official = pd.DataFrame(polemo_official[traintest])

    return df_polemo_official

def load_preprocessed_data(polemo_category: str):
    return pd.read_csv(f"data/{polemo_category}_train_preprocessed.csv", sep=";", header=0)

def remove_word_from_dicts(dicts: dict, word: str):
    """ Remove obvious words from dicts (e.g. "hotel" from opinions about hotels) """
    phrases = [word, f"{word} {word}"]
    for sentiment in dicts:
        for phrase in phrases:
            try:
                dicts[sentiment].remove(phrase)
            except ValueError:
                pass

def remove_shared_words(dicts: dict):
    """ If the word is shared among two or more dictionaries, remove it """
    sentiment = list(dicts.keys())

    for i in range(len(dicts)):
        keywords_to_remove = set()
        for j in range(i+1, len(dicts)):
            dict1 = dicts[sentiment[i]]
            dict2 = dicts[sentiment[j]]
            for keyword in dict1:
                if keyword in dict2:
                    dict2.remove(keyword)
                    keywords_to_remove.add(keyword)  # don't remove yet, because more dicts can share it
        for keyword in keywords_to_remove:
            dict1.remove(keyword)
