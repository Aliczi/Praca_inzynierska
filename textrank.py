from datasets import load_dataset
import pandas as pd
import spacy
import pytextrank

from spacy.matcher import Matcher
from spacy.attrs import POS

# dependencies to_ install:
    # !pip install pytextrank
    # !python -m spacy download pl_corenews_sm


class TextRank:
    def __init__(self, language_model: str):
        self.nlp = spacy.load(language_model) 
        spacy.lang.pl.PolishDefaults.syntax_iterators = {"noun_chunks" : self._get_chunks}  #noun_chunk replacement for polish
        #IDEA: add "hotel to stopwords"
        self.nlp = spacy.load(language_model) # Create language model 
        self.nlp.add_pipe("textrank")

    def _get_chunks(self, doc):
        ### noun_chunks implementation for slovak language
        #TODO: change to polish
        np_label = doc.vocab.strings.add("NP")
        matcher = Matcher(self.nlp.vocab)
        pattern = [{POS: 'ADJ', "OP": "+"}, {POS: {"IN": ["NOUN", "PROPN"]}, "OP": "+"}]
        matcher.add("Adjective(s), (p)noun", [pattern])
        matches = matcher(doc)

        for match_id, start, end in matches:
            yield start, end, np_label  

    def _lemmatize(self, text: str) -> str:
        """ Return lemmatized text"""
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def get_keywords(self, text: str, len=None, lemmatized=False) -> list:
        """ Get keywords from text """
        if lemmatized == True:
            text = self._lemmatize(text)

        doc = self.nlp(text)
        keywords = [phrase.text for phrase in doc._.phrases]
        if len is not None:
            keywords = keywords[:len]
        return keywords
        
    def create_dicts_for_all_classes(self, df: pd.DataFrame, target_colname = "target" , opinion_colname="text", len=25, trainset_size=100) -> dict:
        """ Create dictionaries with keywords for every oppinion class in df. 
            Returns a dictionary in form {class0: [keyword0, keyword1, ...], ...} """
        keywords = dict()
        
        for sentiment in df[target_colname].unique():
            # filter opinions by sentiment
            df_filtered = df[df[target_colname] == sentiment] 
            df_filtered.drop(target_colname, axis=1)

            # get keywords
            df_filtered = df_filtered[:trainset_size] #TODO: DELETE THIS!
            all_oppinions = df_filtered[opinion_colname].sum() # TODO: too long
            keywords[sentiment] = self.get_keywords(all_oppinions, len=len)

        return keywords

def save_dicts_to_files(dicts: dict):
    for sentiment in dicts.keys():
        filename = f"textrank_{sentiment}.txt"
        with open(filename, 'w') as file:
            file.write('\n'.join(dicts[sentiment]))
        print(f"Keywords for sentiment {sentiment} written to file {filename}")

if __name__ == '__main__':
    # ---------------------------------
    # ADJUST PARAMETERS
    
    polemo_category = "hotels_text" # only opinions about hotels
    # available categories: 'all_text', 'all_sentence', 'hotels_text', 'hotels_sentence', 'medicine_text', 'medicine_sentence',
    # 'products_text', 'products_sentence', 'reviews_text', 'reviews_sentence'

    number_of_keywords = 10
    number_of_opinions = 10
    # ----------------------------------
    
    # read data from polemo
    polemo_official = load_dataset("data/polemo2-official/", polemo_category) 
    df_polemo_official = pd.DataFrame(polemo_official["train"])

    # create dictionaries
    textRank = TextRank("pl_core_news_sm")
    dicts = textRank.create_dicts_for_all_classes(df_polemo_official, len=number_of_keywords, trainset_size=number_of_opinions)
    
    # save them
    save_dicts_to_files(dicts)



