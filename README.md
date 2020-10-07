# WordEmbeddings

This repo includes some work exploration I did for fun with word embeddings, including the creation of a new word embedding purely based on semantic/dictionary definitions instead of context words from big corpuses.

## Wiktionary.py
Contains the function necessary to dowanload, analyse and parse data from the Wiktionary, including word definitions and synonyms.

Relevant function includes
- `wiki_clean().unwiki()` to clean xml data of wikitext into readable text without the tags
- `parse_wiktionary_dump()` to process a .xml.bz2 Wiktionary data dump into manageable pickle files
- `Definition()` class for a word token including definitions and synonyms data that will be used in the processed data files

## Semantic.py

Contains code relevant to create word embedding purely based on semantic definitions and synonyms. Relevant function includes
- `import_semantic()` to generate (or import) semantic word embeddings
- `import_branchmark()` to import databases of synonyms evaluation, to evaluate the performance of word embeddings to identify synonyms
- `import_synpairs()` to import all synonym pairs present in the Wiktionary data
- `Embeddings()` class that contains relevant functions to get a word embedding from a token, and evaluate the embeddings based on the previously identified benchmarks

## Translation.py

Contains code relevant to translate word embeddings from an origin language to a destination language, based on anther multilingual embeddings mapping those languages to the same latent space. Only one relevant function:
- `translate()`

This code uses a pre-trained map of multilingual embeddings (e.g. Glove) to translate another embedding space (e.g. Paragrams)

It does so by projecting a vector in the target language (e.g. French) in the orthonormal basis spanned by the (ordered) Gram-Schmidt decomposition of the K (<= n_dim) nearest neighbours in the source language (e.g. English)

It then creates another orthonormal basis spanned by the (ordered) Gram-Schmidt decomposition of the same tokens in the source language e.g. English), but in the new embeddings space

With this new basis, it recreates the target language (e.g. French) vector in the new embedding space

note: the K nearest neighbours are measured based on CSLS (Cross-Lingual Similarity Scaling) distance from "Word translation without parallel data", Conneau et al., ICLR 2018

This process supposes two things:

a) the K (<= n_dim) nearest neighbours in multilingual embeddings space is enough to construct an accurate translation of the token

b) The cosine distance between vectors from different languages in multilingual embeddings is preserved between embedding space (e.g. between Glove and Paragrams)

In other words, the relationships between "cat" and "tiger" can differ between embedding spaces (Glove might put them far because they are used in different contexts, but Paragrams might put them close cause they are both felines), but what matters is that the relationships between "chat" and "cat" (and "tigre" and "tiger") is the same in both multilingual space (both Glove and Paragrams would put them very close cause they are
translations of each others)

The limit of this supposition is complex translations (e.g. "feuille" is gonna be close to both "sheet" and "leaf", but the relationship between "sheet" and "leaf" will vary between embedding spaces, so this will not respect axiom b)

# Library dependancies

```
bzip2 = 1.0.8
numpy = 1.18.5
scipy = 1.5.0
tqdm = 4.32.2
unidecode = 1.1.1
pytorch = 1.3.0
```


# Known bug

The regex in the Wiktionary library to *unwiki()* the text can't handle nested tags (i.e. `{{l|en|{{m|en|word}}}}`). It needs to be rewritten and I'm still bad with regex.

# To-do

- Comparison testing on modern pretrained embedding needs to be done.

- Checking the efficiency on corpus size (for rare languages or rare words embeddings where massive corpus pretraining might not be a possibility) by comparing to embeddings trained on similar sized corpuses.

- Checking the efficiency of definitions only to account for synonym testing data. For now, synonyms are included in the embedding training through their inclusion in *def_matrix*, and are also the evaluation test, so there's definite data pollution for the implemented metrics.

- Checking the efficiency of vocabulary limitations. Performance degraded when no limitations occurred, which should not be the case.

- Weighing the weights of the word definitions in *def_matrix* did lead to better results with the benchmarks. A technique that worked was to run dependancy parsing on definitions, giving a "rank" to each word token based on how far they were from the sentence's root, and multiplying the def_matrix element by 1/(rank+C), C being a constant (C=2 worked well).

```
# Download from https://stanfordnlp.github.io/CoreNLP/download.html
from nltk.parse.stanford import StanfordDependencyParser

path_to_jar = 'stanford-corenlp-full-2018-02-27\\stanford-corenlp-3.9.1.jar'
path_to_models_jar = 'stanford-corenlp-full-2018-02-27\\stanford-corenlp-3.9.1-models.jar'
os.environ['JAVAHOME'] = 'C:\\Program Files (x86)\\Java\\jre8\\bin\\java.exe'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

for t in next(dependency_parser.raw_parse(definition_string)).triples():
  # DO SOMETHING
```

- Using nltk part-of-speech tagging in definitions instead of *wiktionary().main_pos()* might lead to more accurate part-of-speech tagging. Here is some code for it.

```
from nltk import pos_tag, word_tokenize
pos_data = pos_tag(word_tokenize(defin))
```
