import pandas as pd
from noun_chunks_pl import noun_chunks_pl
import spacy
import pytextrank
from tools import *

# dependencies to_ install
# !pip install pytextrank
# !python -m spacy download pl_corenews_sm


class TextRank:
    def __init__(self, language_model: str, textrank_type: str = "textrank"):
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

        doc = self.nlp(text)
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
        trainset_size: int = None, # use only for development purposes – shortens computation
    ) -> dict:
        """ Create dictionaries with keywords for every oppinion class in df.
        Returns a dictionary in form {class0: [keyword0, keyword1, ...], ...} """
        keywords = dict()

        if trainset_size:
            df_filtered = df[:trainset_size]
        else:
            df_filtered = df
        df_filtered = df_filtered.groupby(target_colname).sum(opinion_colname)
        keywords = dict(df_filtered[opinion_colname].apply(self._get_keywords, len=len))
        
        return keywords



if __name__ == "__main__":
    # --------------------------------------------------------------------------
    number_of_keywords = 100
    number_of_opinions = 100
    textrank_type = "textrank"  #textrank, positionrank, topicrank, biasedtextrank

    polemo_category = "hotels_text"  # only opinions about hotels
    # available categories: 'all_text', 'all_sentence',
    # 'hotels_text', 'hotels_sentence', 'medicine_text', 'medicine_sentence',
    # 'products_text', 'products_sentence', 'reviews_text', 'reviews_sentence'
    # --------------------------------------------------------------------------

    # df_polemo_official = load_raw_data("data/polemo2-official/", polemo_category)
    df_polemo_official = load_preprocessed_data(polemo_category)

    textRank = TextRank("pl_core_news_sm", textrank_type)
    dicts = textRank.create_dicts_for_all_classes(
        df_polemo_official,
        len=number_of_keywords,
        trainset_size=number_of_opinions
    )

    remove_word_from_dicts(dicts, "hotel")
    remove_shared_words(dicts)

    save_dicts_to_files(dicts, textrank_type)

