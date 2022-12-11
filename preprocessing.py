import sys
sys.path.insert(0, './spellcorrectorpl/python')
# from KnownWordsProvider import KnownWordsProviderUsingRAM, KnownWordsProviderUsingBigFile, KnownWordsProviderUsingMultipleFiles
from BigramsProvider import BigramsProvider
from SpellCorrector import SpellCorrector

from tools import load_raw_data

from string import punctuation
import pandas as pd
import re
import unidecode
import spacy

UNIGRAMS_FILEPATH = 'spellcorrectorpl/out/1grams_fixed'
UNIGRAMS_FILES_DIR = 'spellcorrectorpl/out/1grams_splitted/'
BIGRAMS_FILEPATH = "spellcorrectorpl/out/2grams_splitted"


def create_corrector():
    words_provider = KnownWordsProviderUsingMultipleFiles()
    bigrams_provider = BigramsProvider()
    words_provider.initialize(UNIGRAMS_FILES_DIR)
    bigrams_provider.initialize(BIGRAMS_FILEPATH)

    return SpellCorrector(words_provider, bigrams_provider)

def correct_opinion(opinion: str, corrector: SpellCorrector):        
    opinion = re.sub(r'\d', "<liczba>", opinion)  #change numbers to <liczba>
    opinion = corrector.sentence_correction(opinion, print_words=False)
    return opinion

def lemmatize(opinion: str, nlp) -> str:
    opinion = re.sub(' +', ' ', opinion) # remove multiple spaces
    doc = nlp(opinion)
    return " ".join([token.lemma_ for token in doc])

def preprocess(df: pd.DataFrame):
    nlp = spacy.load("pl_core_news_sm")
    translate_table = dict((ord(char), None) for char in punctuation)
    oppinions = df.copy()
    oppinions["text"] = oppinions["text"].str.translate(translate_table)  # remove punctuation
    # TODO: usuwać słowa w cudzysłowie?

    # corrector = create_corrector()  
    # oppinions["text"] = oppinions["text"].apply(correct_opinion, corrector=corrector)

    oppinions["text"] = oppinions["text"].str.lower()  # to lower
    oppinions["text"] = oppinions["text"].apply(lemmatize, nlp=nlp)
    # oppinions["text"] = oppinions["text"].apply(unidecode.unidecode)  # remove polish characters
  
    return oppinions


if __name__=="__main__":
    # available categories for polemo: 'all_text', 'all_sentence',
    # 'hotels_text', 'hotels_sentence', 'medicine_text', 'medicine_sentence',
    # 'products_text', 'products_sentence', 'reviews_text', 'reviews_sentence'
    # --------------------------------------------------------------------------
    polemo_category = "medicine_text"
    train_or_test = "train"
    df_polemo_official = load_raw_data("data/polemo2-official/", polemo_category, train_or_test)

    oppinions = preprocess(df_polemo_official)
    oppinions.to_csv(f"data/{polemo_category}_{train_or_test}_preprocessed.csv", sep=';', index=False)

