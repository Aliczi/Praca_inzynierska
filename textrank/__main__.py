from audioop import bias
from collections import Counter
from telnetlib import X3PAD
from sklearn.feature_extraction.text import CountVectorizer
import sys
import pandas as pd
from textrank.noun_chunks_pl import noun_chunks_pl
import spacy
import pytextrank
from tools import *

# dependencies to_ install
# !pip install pytextrank
# !python -m spacy download pl_core_news_sm TODO zmienić w readme na python -m textrank, bo inaczej nie dziala


class TextRank:
    bias_words = None

    def __init__(self, language_model: str, textrank_type: str = "textrank"):
        self.nlp = spacy.load(language_model)
        spacy.lang.pl.PolishDefaults.syntax_iterators = {
            "noun_chunks": self._get_chunks
        }  # noun_chunk replacement for polish
        self.nlp = spacy.load(language_model)

        self.nlp.add_pipe(textrank_type)
        self.nlp.max_length = 1100000

    def _get_chunks(self, doc):
        return noun_chunks_pl(doc)

    def _get_keywords(
        self, text: str, len: int = None, sentiment: int = None
    ) -> list:
        """Get keywords from text"""
        doc = self.nlp(text)
        if self.bias_words:
            doc._.textrank.change_focus(focus=self.bias_words[sentiment], bias=1.0, default_bias=0)
        keywords = [phrase.text for phrase in doc._.phrases]
        if len is not None:
            keywords = keywords[:len]
        return keywords

    def _generate_bias_words(self, n: int, df: pd.DataFrame, grouped_text):
        self.bias_words = {}

        for target in df["target"].unique():
            targeted = grouped_text.get_group(target)["text"].values
            # Extract candidate words/phrases
            count = CountVectorizer(lowercase=True, min_df=0.015, max_features=n, stop_words=self.nlp.Defaults.stop_words)\
                    .fit(targeted)
            self.bias_words[target] = " ".join(count.get_feature_names_out())

    def create_dicts_for_all_classes(
        self,
        df: pd.DataFrame,
        target_colname: str = "target",
        opinion_colname: str = "text",
        len: int = 25,  # maximum length of the keywords list
        trainset_size: int = None,  # use only for development purposes – shortens computation
        join_oppinions: bool = True,  # create a long text from all oppinions with the same sentiment
        bias_context_len: int = None,  # number of words generated for bias context
    ) -> dict:
        """ Create dictionaries with keywords for every oppinion class in df.
        Returns a dictionary in form {class0: [keyword0, keyword1, ...], ...} """

        if trainset_size:
            df_filtered = df[1:trainset_size]
        else:
            df_filtered = df.copy()

        grouped = df_filtered.groupby(target_colname)

        if "biasedtextrank" in self.nlp.pipe_names:
            self._generate_bias_words(bias_context_len, df, grouped)

        keywords = {}
        if join_oppinions:
            # join all oppinions in one text
            df_filtered = grouped.sum(numeric_only=False)
            for sentiment in df[target_colname].unique():
                keywords[sentiment] = self._get_keywords(
                    df_filtered[opinion_colname][sentiment],
                    len=len,
                    sentiment=sentiment
                )

        else:
            # create a dict for each oppinion and join them
            for sentiment in df[target_colname].unique():
                df_filtered = df[df[target_colname] == sentiment]
                this_sentiment_keywords = []

                for opinion in df_filtered[opinion_colname]:
                    opinion_keywords = self._get_keywords(opinion, len=len, sentiment=sentiment)
                    this_sentiment_keywords += opinion_keywords
                # get keywords which appear in the most oppinions
                ct = Counter(this_sentiment_keywords)
                keywords[sentiment] = list(dict(sorted(dict(ct).items(), key=lambda item: item[1], reverse=True)).keys())[:len]

        return keywords


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    max_number_of_keywords = 4000
    number_of_opinions = None
    join_oppinions = True
    textrank_type = "all"  # textrank, positionrank, topicrank, biasedtextrank
    bias_context_len = 20  # length of the list generated for bias textrank with count vectorize
    preprocessed = "all"
    polemo_category = "all_text"  # only opinions about hotels
    # available categories: 'all_text', 'all_sentence',
    # 'hotels_text', 'hotels_sentence', 'medicine_text', 'medicine_sentence',
    # 'products_text', 'products_sentence', 'reviews_text', 'reviews_sentence'
    # --------------------------------------------------------------------------
    if (polemo_category == "all_text"):
        polemo_categories = ["hotels_text", "medicine_text", "products_text", "reviews_text"]
    else:
        polemo_categories = [polemo_category]

    if (preprocessed == "all"):
        preprocessing_categories = ["simple", "lemmatization"]  # TODO add 'spelling' when done
    else:
        preprocessing_categories = [preprocessed]
    if textrank_type == "all":
        models = ["textrank", "positionrank"] #TODO topicrank/biasedtextrank problems
    else:
        models = [textrank_type]
    if (preprocessed != "none"):
        for category in polemo_categories:
            for preprocessing_category in preprocessing_categories:
                data = load_preprocessed_data(category, preprocessing_category)
                for model in models:
                    print(f"{category}_{preprocessing_category}_textrank_{model}_{max_number_of_keywords} beginning!")
                    textRank = TextRank("pl_core_news_sm", model)
                    dicts = textRank.create_dicts_for_all_classes(
                        data,
                        len=max_number_of_keywords,
                        trainset_size=number_of_opinions,
                        join_oppinions=join_oppinions,
                        bias_context_len=bias_context_len
                    )
                    save_dicts_to_files(dicts, f"{category}_{preprocessing_category}_textrank_{model}_{max_number_of_keywords}", "out/textrank")
                    print(f"{category}_{preprocessing_category}_textrank_{model}_{max_number_of_keywords} done!")
    else:
        df_polemo_official = load_raw_data("data/polemo2-official/", polemo_category)
        print(df_polemo_official.head())
        # df_polemo_official = load_preprocessed_data(polemo_category)

        textRank = TextRank("pl_core_news_sm", textrank_type)
        dicts = textRank.create_dicts_for_all_classes(
            df_polemo_official,
            len=max_number_of_keywords,
            trainset_size=number_of_opinions,
            join_oppinions=join_oppinions,
            bias_context_len=bias_context_len
        )

        remove_word_from_dicts(dicts, "hotel")
        remove_shared_words(dicts)

        save_dicts_to_files(dicts, textrank_type + ("_joined" if join_oppinions else "_sep"))
