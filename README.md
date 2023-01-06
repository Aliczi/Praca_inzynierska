# Praca inżynierska

## usage

### textrank
`textrank` -- provide the parameters in the code, from the main dir run:
```sh
python3 -m textrank
```    

 number_of_keywords = 100
    number_of_opinions = None
    join_oppinions = False

You can fint the example run of the textrank in `textrank/__main__.py`.

To run TextRank, **create an instance of `TextRank`** class first:

```python
textRank = TextRank("pl_core_news_sm", textrank_type)
```
where:

`pl_core_news_sm` is the language model to be read by spacy

`textrank_type` is textrank, positionrank, topicrank or biasedtextrank (you can read about it further in our engeneer thesis)

If you choose biasedtextrank you have to provide third argument – `bias_words` i.e. words which will be used as a bias context, separated by spaces.

Then you can **create dictionaries** for all classes in your dataset:

```python
dicts = textRank.create_dicts_for_all_classes(df)
```
`df` is a pandas dataframe with your data. Data should have columns with oppinion and its category.

You can pass the name of the oppinion column via `opinion_colname` and the category column via `target_colname`. By default they're "text" and "target".

Optionally you can determine maximum length of the keywords list using parameter `len`.

You can also choose whether the oppinions should be joined or not (`join_oppinions`).
If this parameter is set to `True`, we create a long text from all oppinions with the same sentiment first and then it builds the graph and dictionaries. If it's set to `False`, we create a dictionary for each oppinion and then join them based on their sentiment.

### tf-idf
`tf-idf` -- provide the parameters in the code, from the main dir run:
```sh
python3 tf-idf.py
```

### sbert
`sbert` -- provide parameters from command line,by default polemo_category=hotels, model= distilbert-base-nli-mean-tokens.
From the main dir run:
```sh
python sbert.py --polemo_category=reviews_text --model=sentence-transformers/distiluse-base-multilingual-cased-v1
```
For help run:
```sh
python3 sbert.py -h
```

## Useful functions

In `tools.py` you can find functions:

`load_raw_data(data_path, data_category, traintest)`

where:

`data_path` can be eg. "data/polemo2-official/"

`polemo_category` can be 'all_text', 'all_sentence', 'hotels_text', 'hotels_sentence', 'medicine_text', 'medicine_sentence', 'products_text', 'products_sentence', 'reviews_text' or 'reviews_sentence'</br>
`traintest` determines split you want to use ('train', 'test' or 'validation')

`load_preprocessed_data(polemo_category)` – reads preprocessed data from the f"data/{polemo_category}_train_preprocessed.csv" directory.

`remove_word_from_dicts(dicts, word)` – which removes the `word` from all dictionaries in `dicts` which is a python dictionary with sentiment as a key and list of keywords for it as a value.

`remove_shared_words(dicts)` – removes words which appear in more than one dictionary.

`save_dicts_to_files(dicts, prefix)` – saves dictionaries to the files named f"out/{prefix}_{sentiment}.txt"


## Installation
To install Sentence-Transformer please follow these notes https://www.sbert.net/docs/installation.html
