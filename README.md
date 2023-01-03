# Praca inżynierska

### usage
`textrank` -- provide the parameters in the code, from the main dir run:
```sh
python3 -m textrank
```

`tf-idf` -- provide the parameters in the code, from the main dir run:
```sh
python3 tf-idf.py
```

`sbert` -- provide parameters from command line,by default polemo_category=hotels, model= distilbert-base-nli-mean-tokens.
From the main dir run:
```sh
python sbert.py --polemo_category=reviews_text --model=sentence-transformers/distiluse-base-multilingual-cased-v1
```
For help run:
```sh
python3 sbert.py -h
```


### Installation
To install Sentence-Transformer please follow these notes https://www.sbert.net/docs/installation.html
