import pandas as pd
from datasets import load_dataset


def save_dicts_to_files(dicts: dict, prefix: str):
    for sentiment in dicts:
        filename = f"{prefix}_{sentiment}.txt"
        with open(filename, "w") as file:
            file.write("\n".join(dicts[sentiment]))
        print(f"Keywords for sentiment {sentiment} written to file {filename}")


def load_raw_data(data_path: str, data_category: str):
    polemo_official = load_dataset(data_path, data_category)
    df_polemo_official = pd.DataFrame(polemo_official["train"])

    return df_polemo_official

def load_preprocessed_data(data_path: str):
    return pd.read_csv(data_path, sep=";", header=0)
