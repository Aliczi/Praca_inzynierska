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
    parser.add_argument("--model", default='sentence-transformers/distiluse-base-multilingual-cased-v1',
                            help="Other model: microsoft/Multilingual-MiniLM-L12-H384, sentence-transformers/all-distilroberta-v1" )
    parser.add_argument("--preprocessed", default="none",
                        help="Choosing type of preprocessing. Available options: 'none', 'all', 'simple', 'lemmatization', 'spelling'") #TODO add note to readme file
    args = parser.parse_args()

    if(args.polemo_category == "all_text"):
        polemo_categories = ["hotels_text", "medicine_text", "products_text", "reviews_text"]
    else:
        polemo_categories = [args.polemo_category]

    if(args.preprocessed == "all"):
        preprocessing_categories = ["simple", "lemmatization"] #TODO add 'spelling' when done
    else:
        preprocessing_categories = [args.preprocessed]

    model = SentenceTransformer(args.model)

    if(args.preprocessed != "none"):
        for category in polemo_categories:
            for preprocessing_category in preprocessing_categories:
                data = load_preprocessed_data(category,preprocessing_category)
                keywords_dic = sbert(data, model)
                save_dicts_to_files(keywords_dic, f"{category}_{preprocessing_category}_sbert", "out")
                print(f"{category}_{preprocessing_category}_sbert done!")
    else:
        data = load_raw_data("data/polemo2-official/", args.polemo_category)
        keywords_dic = sbert(data, model)
        remove_word_from_dicts(keywords_dic, "hotel")
        remove_shared_words(keywords_dic)
        save_dicts_to_files(keywords_dic, "sbert", "out")

    sys.exit()
    
