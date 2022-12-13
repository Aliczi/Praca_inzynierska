from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import sys

from tools import *

polemo_category = "hotels_text"
df = load_raw_data("data/polemo2-official/", polemo_category)

grouped = df.groupby("target")
model = SentenceTransformer('distilbert-base-nli-mean-tokens')


import re
def remove(list):
    pattern = '[0-9,.)()]'
    list = [re.sub(pattern, '', i) for i in list]
    return list


def count_vec_candidates(doc: str):
    n_gram_range = (1, 2) #unigrams and bigrams
    
    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range,lowercase=True, min_df=0.015).fit(doc)
    return count.get_feature_names_out()



def main():
    for target in df["target"].unique():
        targeted = grouped.get_group(target)["text"].values
        t = list(targeted[:1])
        t= remove(t)

        candidates = count_vec_candidates(t)
        print("Potential number of keyphrases: ", len(candidates))
        doc_embedding = np.mean(model.encode(t), axis=0)  #5 mins for  1200 docs
        print("Doc embedding {}".format(target))
        candidate_embeddings = model.encode(candidates)

        top_n = 100
        distances = cosine_similarity([doc_embedding], candidate_embeddings)
        keywords = [(candidates[index], distances[0][index]) for index in distances.argsort()[0][-top_n:]]

        kk = pd.DataFrame(keywords, columns=["word", "sbert"])
        keywords_dic[target] = list(kk["word"])


if __name__=='__main__':
    keywords_dic = dict()
    main()

    remove_word_from_dicts(keywords_dic, "hotel")
    remove_shared_words(keywords_dic)
    
    save_dicts_to_files(keywords_dic, "sbert")
    sys.exit() 
    
