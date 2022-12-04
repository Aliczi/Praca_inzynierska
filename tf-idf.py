from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tools import load_data, save_dicts_to_files

n_keywords = 50
df = load_data("data/polemo2-official/", "hotels_text")

vectorizer = TfidfVectorizer()

grouped = df.groupby("target")
keywords = dict()
for target in df["target"].unique():
    targeted = grouped.get_group(target)["text"].values
    tfidf_vector = vectorizer.fit_transform(targeted)
    tf_idf_results = pd.DataFrame(
        zip(*[vectorizer.get_feature_names_out(), vectorizer.idf_]),
        columns=["word", "tf-idf"]
        ).sort_values("tf-idf",  ascending=False)
    keywords[target] = tf_idf_results["word"][:n_keywords]

save_dicts_to_files(keywords, "tfidf")
