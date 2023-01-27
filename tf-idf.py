from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tools import *

def tfidf(df: pd.DataFrame, n_keywords: int):
    vectorizer = TfidfVectorizer()

    grouped = df.groupby("target")
    keywords = dict()
    for target in df["target"].unique():
        targeted = grouped.get_group(target)["text"].values
        tfidf_matrix = vectorizer.fit_transform(targeted)
        tfidf_matrix_dataFrame = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix).mean(axis = 0).to_frame()
        tfidf_matrix_dataFrame['id'] = tfidf_matrix_dataFrame.index
        word_id = pd.DataFrame(vectorizer.vocabulary_.items(), columns = ["word", "id"])
        
        answer = pd.merge(tfidf_matrix_dataFrame, word_id, how="inner", on=["id"])

        answer = answer.drop(columns=['id'], axis=1) \
            .rename(columns={0: "mean_tf_idf"}) \
            .sort_values("mean_tf_idf",  ascending=False) 

        # df = pd.concat([tfidf_matrix_dataFrame, df4], axis=1).
        # tf_idf_results = pd.DataFrame(
        #     zip(*[vectorizer.get_feature_names_out(), vectorizer.idf_]),
        #     columns=["word", "tf-idf"]
        #     ).sort_values("tf-idf",  ascending=False)
        keywords[target] = list(answer["word"][:n_keywords])
    return keywords


if __name__ == '__main__':
    n_keywords = 100
    polemo_category = "hotels_text"
    df = load_raw_data("data/polemo2-official/", polemo_category)
    # df = load_preprocessed_data(polemo_category)
    keywords = tfidf(df, n_keywords)
    # remove_word_from_dicts(keywords, "hotel")
    # remove_shared_words(keywords)

    save_dicts_to_files(keywords, "tfidf")
