import sys
import pandas as pd
from textrank.noun_chunks_pl import noun_chunks_pl
import spacy
import pytextrank
from tools import *

# dependencies to_ install
# !pip install pytextrank
# !python -m spacy download pl_corenews_sm


class TextRank:
    def __init__(self, language_model: str, textrank_type: str = "textrank", bias_words: str = None):
        self.bias_words = bias_words
        self.nlp = spacy.load(language_model)
        spacy.lang.pl.PolishDefaults.syntax_iterators = {
            "noun_chunks": self._get_chunks
        }  # noun_chunk replacement for polish
        self.nlp = spacy.load(language_model)

        self.nlp.add_pipe(textrank_type)

    def _get_chunks(self, doc):
        return noun_chunks_pl(doc)


    def _get_keywords(
        self, text: str, len=None
    ) -> list:
        """Get keywords from text"""
        print(text)
        doc = self.nlp(text)
        if bias_words != None:
            doc._.textrank.change_focus(focus=self.bias_words, bias=1.0, default_bias=0)
        keywords = [phrase.text for phrase in doc._.phrases]
        if len is not None:
            keywords = keywords[:len]
        return keywords

    def create_dicts_for_all_classes(
        self,
        df: pd.DataFrame,
        target_colname: str = "target",
        opinion_colname: str = "text",
        len: int = 25,
        trainset_size: int = None # use only for development purposes – shortens computation
    ) -> dict:
        """ Create dictionaries with keywords for every oppinion class in df.
        Returns a dictionary in form {class0: [keyword0, keyword1, ...], ...} """
        keywords = dict()

        if trainset_size:
            df_filtered = df[1:trainset_size]
        else:
            df_filtered = df
        df_filtered = df_filtered.groupby(target_colname).sum(opinion_colname)
        keywords = dict(df_filtered[opinion_colname].apply(self._get_keywords, len=len))
        
        return keywords



if __name__ == "__main__":
    # --------------------------------------------------------------------------
    number_of_keywords = 100
    number_of_opinions = None
    textrank_type = "textrank"  # textrank, positionrank, topicrank, biasedtextrank
    words_for_bias = ["hotel", "pokój", "łazienka", "polecenie", "dobry", "straszny", "zły", "ładny"]
    polemo_category = "hotels_text"  # only opinions about hotels
    # available categories: 'all_text', 'all_sentence',
    # 'hotels_text', 'hotels_sentence', 'medicine_text', 'medicine_sentence',
    # 'products_text', 'products_sentence', 'reviews_text', 'reviews_sentence'
    # --------------------------------------------------------------------------
   
    bias_words = None if textrank_type != "biasedtextrank" else " ".join(words_for_bias)
   
    # df_polemo_official = load_raw_data("data/polemo2-official/", polemo_category)
    df_polemo_official = load_preprocessed_data(polemo_category)

    textRank = TextRank("pl_core_news_sm", textrank_type, bias_words)
    dicts = textRank.create_dicts_for_all_classes(
        df_polemo_official,
        len=number_of_keywords,
        trainset_size=number_of_opinions
    )

    remove_word_from_dicts(dicts, "hotel")
    remove_shared_words(dicts)

    save_dicts_to_files(dicts, textrank_type)

