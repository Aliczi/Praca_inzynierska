import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import sys
from tools import *



import re
def preprocess(list):
    pattern = '[0-9,.)()]'
    list = [re.sub(pattern, '', i) for i in list]
    return list


def count_vec_candidates(doc: str):
    n_gram_range = (1, 2) #unigrams and bigrams
    
    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range,lowercase=True, min_df=0.015).fit(doc)
    return count.get_feature_names_out()


def sbert(data: pd.DataFrame, model: SentenceTransformer):
    grouped = data.groupby("target")
    keywords_dic = dict()
    for target in data["target"].unique():
        targeted = grouped.get_group(target)["text"].values
        t= preprocess(list(targeted))

        candidates = count_vec_candidates(t)
        print("Potential number of keyphrases: ", len(candidates))
        doc_embedding = np.mean(model.encode(t), axis=0)  #5 mins for  1200 docs
        print("Doc embedding for: {}".format(target))
        candidate_embeddings = model.encode(candidates)

        top_n = 100
        distances = cosine_similarity([doc_embedding], candidate_embeddings)
        keywords = [candidates[index] for index in distances.argsort()[0][::-1][:top_n]]
        
        keywords_dic[target] = keywords

    return keywords_dic


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    description = 'Keyword extraction using Sentence-BERT')
    parser.add_argument("--polemo_category", default="hotels_text",
                            help="Other available: 'all_text', 'all_sentence', 'hotels_text', 'hotels_sentence', 'medicine_text', 'medicine_sentence', 'products_text', 'products_sentence', 'reviews_text', 'reviews_sentence'")
    parser.add_argument("--model", default='distilbert-base-nli-mean-tokens',
                            help="Other model: sentence-transformers/distiluse-base-multilingual-cased-v1, sentence-transformers/all-distilroberta-v1" )
    args = parser.parse_args()


    data = load_raw_data("data/polemo2-official/", args.polemo_category)
    model = SentenceTransformer(args.model)          
    keywords_dic = sbert(data, model)

    remove_word_from_dicts(keywords_dic, "hotel")
    remove_shared_words(keywords_dic)

    save_dicts_to_files(keywords_dic, "sbert")
    sys.exit() 
    
