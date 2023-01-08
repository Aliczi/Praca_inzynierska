from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tools import *

def tfidf(df: pd.DataFrame, n_keywords: int):
    vectorizer = TfidfVectorizer()

    grouped = df.groupby("target")
    keywords = dict()
    for target in df["target"].unique():
        targeted = grouped.get_group(target)["text"].values
        vectorizer.fit_transform(targeted)
        tf_idf_results = pd.DataFrame(
            zip(*[vectorizer.get_feature_names_out(), vectorizer.idf_]),
            columns=["word", "tf-idf"]
            ).sort_values("tf-idf",  ascending=False)
        keywords[target] = list(tf_idf_results["word"][:n_keywords])
    return keywords


if __name__ == '__main__':
    n_keywords = 100
    polemo_category = "hotels_text"
    # df = load_raw_data("data/polemo2-official/", polemo_category)
    df = load_preprocessed_data(polemo_category)
    keywords = tfidf(df, n_keywords)
    remove_word_from_dicts(keywords, "hotel")
    remove_shared_words(keywords)

    save_dicts_to_files(keywords, "tidf")
