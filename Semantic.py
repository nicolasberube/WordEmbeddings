#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:53:54 2020

@author: berube
"""
import os
from pathlib import Path
from datetime import datetime
import pickle
import time
from tqdm import tqdm
from scipy.stats import spearmanr
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
import numpy as np
# from nltk.parse.stanford import StanfordDependencyParser
from collections import Counter
# Homemade Wiktionary function
from Wiktionary import Definition, wiktionary
from unidecode import unidecode
import matplotlib.pyplot as plt


def tokenize(passage):
    """Returns list of word tokens in a sentence.

    Parameters
    ----------
    passage: str
        The sentence to clean

    Returns
    -------
    list of str
        The list of lowercase word tokens.
    """
    newpassage = ''
    # Iteration over individual characters
    for i, c in enumerate(passage):
        if c.isalnum():
            newpassage += c
        # Keeping punctuation in numbered data and time
        elif (c in '.,:' and
              i > 0 and
              i < len(passage)-1 and
              passage[i-1].isnumeric() and
              passage[i+1].isnumeric()):
            newpassage += c
        # Keeping the apostrophes and hyphens in words
        elif (c in '\'-' and
              i > 0 and
              i < len(passage)-1 and
              passage[i-1].isalpha() and
              passage[i+1].isalpha()):
            newpassage += c
        else:
            newpassage += ' '
    return newpassage.split()


def import_benchmark(b_path="data//Benchmarks"):
    """Imports benchmarking word pair files.

    The following files need to be in the appropriate directory:

    The MEN Test Collection
    https://staff.fnwi.uva.nl/e.bruni/MEN

    SimLex-999
    https://fh295.github.io/simlex.html

    SimVerb-3500
    https://github.com/JoonyoungYi/datasets/tree/master/simverb3500

    Standford Rare Word (RW) Similarity dataset
    https://nlp.stanford.edu/~lmthang/morphoNLM/

    WordSim-353
    http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.html

    MTurk-287
    https://github.com/
        mfaruqui/word-vector-demo/blob/master/data/EN-MTurk-287.txt

    MTurk-771
    http://www2.mta.ac.il/~gideon/mturk771.html

    Parameters
    ----------
    b_path : Path, optional
        Path of the benchmark file folders described previously.
        Default is 'data/Benchmarks'

    Returns
    -------
    dict of str:list of tuples
        Dict object where key=(str) the name of the dataset and
        value=similarity data of the dataset
        Each similarity data those is a list of tuple
        (str, str, float, str, str)
        which are (Word1, Word2, similarity between words between 0 and 1,
                   part-of-speech of Word1, part-of-speech of Word2)
        It will contain 7 lists of the respective 7 datasets pair of words:
            [MEN, SimLex-999, SimVerb-3500, RW, WordSim-353,
             MTurk-287, MTurk-771]
        Part-of-speech are only available for MEN, SimLex-999, SimVerb-3500.
        For the other datasets, it's only a list of 3-tuples (str, str, float)
    """

    b_path = Path(b_path)

    print(str(datetime.now())+'\t'+'Importing benchmark file')
    # J: Adjective, V:Verb, N:Noun

    with open(b_path / 'MEN' / 'MEN_dataset_lemma_form_full') as f:
        MEN_data = sorted([[lin.split()[0][:-2],
                            lin.split()[1][:-2],
                            2/100*float(lin.split()[2]),
                            lin.split()[0][-1:].upper(),
                            lin.split()[1][-1:].upper()]
                           for lin in f.readlines()],
                          key=lambda k: -k[2])

    with open(b_path / 'SimLex-999' / 'SimLex-999.txt') as f:
        f.readline()
        SimLex_data = sorted([[lin.split()[0],
                               lin.split()[1],
                               10/100*float(lin.split()[3]),
                               lin.split()[2].replace('A', 'J'),
                               lin.split()[2].replace('A', 'J')]
                              for lin in f.readlines()],
                             key=lambda k: -k[2])

    with open(b_path / 'SimVerb3500' / 'SimVerb-3500.txt') as f:
        SimVerb_data = sorted([[lin.split()[0],
                                lin.split()[1],
                                10/100*float(lin.split()[3]),
                                lin.split()[2],
                                lin.split()[2]]
                               for lin in f.readlines()],
                              key=lambda k: -k[2])

    with open(b_path / 'rw' / 'rw.txt') as f:
        RW_data = sorted([[lin.split()[0],
                           lin.split()[1],
                           10/100*float(lin.split()[2])]
                          for lin in f.readlines()],
                         key=lambda k: -k[2])

    with open(b_path / 'WordSim353' / 'combined.tab') as f:
        f.readline()
        WordSim_data = sorted([[lin.split()[0],
                                lin.split()[1],
                                10/100*float(lin.split()[2])]
                               for lin in f.readlines()],
                              key=lambda k: -k[2])

    with open(b_path / 'MTurk' / 'MTurk-287.txt') as f:
        MTurk_data = sorted([[lin.split()[0],
                              lin.split()[1],
                              25/100*(float(lin.split()[2])-1)]
                             for lin in f.readlines()],
                            key=lambda k: -k[2])

    with open(b_path / 'MTurk' / 'MTurk-771.csv') as f:
        MTurk2_data = sorted([[lin.replace(',', '\t').split()[0],
                               lin.replace(',', '\t').split()[1],
                               25/100*(float(
                                   lin.replace(',', '\t').split()[2]
                                   )-1)]
                              for lin in f.readlines()],
                             key=lambda k: -k[2])

    all_pairs = {'MEN': MEN_data,
                 'SimLex-999': SimLex_data,
                 'SimVerb-3500': SimVerb_data,
                 'StanfordRW': RW_data,
                 'WordSim-353': WordSim_data,
                 'MTurk-287': MTurk_data,
                 'MTurk-771': MTurk2_data}

    return all_pairs


def import_semantic(embedding_dim=300,
                    vocab_limit=0,
                    lang='english',
                    pos_list={'noun', 'verb', 'adjective'}):
    """Imports (or generates) semantic word embeddings.

    It will use (or save) 3 files from the ./data/ folder:
    idx_words_to_matidx.pkl:
        a dictionary where  key=(int) the index of a word token in the
        Wiktionary file obtaines with the function wiktionary()
        and value=(int) the index column of the objects def_matrix.npz
        and def_embeddings.npy corresponding to the specified
        word token

    def_matrix.npz:
        a symmetrical sparse matrix where each element (i,j) represents
        the similarity between word tokens i and j based on
        the presence of a token in another's definition of synonym list.
        This is a temporary file for the calculation and is technically
        not used per se, but can offer insight.

    def_embeddings.npy:
        the embedding matrix, where each column is the embedding of the
        word token

    Each object file name has a suffix. The first part is the vocabulary
    (or matrix size) limitation, which corresponds to the parameter
    vocab_limit. Suffix is of the form "V{vocab_limit}". Of there is
    no vocabulary limitations, this first suffix is "full"

    The second suffix is the dimension of the embedding, which corresponds
    to the parameter embedding_dim. Suffix is of the form "D{embedding_dim}"

    Parameters
    ----------
    embedding_dim: int, optional
        Dimension of the embedding. Default is 300.

    vocab_limit: int, optional
        Limitation of the vocabulary to the top vocab_limit words based
        on occurences in definitions and synonyms. If 0, then no limitation
        will be done. Default is 0.

    lang: str, optional
        Language to consider for the embedding. Must correspond to a
        key in the Definition() object from the Wiktionary input file
        from wiktionary() function.
        Default is 'english'

    pos_list: dict of str, optional
        Parts-of-speech to consider for the word tokens. Every other
        part-of-speech will be completely ignored. Must correspond to a
        key in the Definition() object from the Wiktionary input file
        from wiktionary() function.
        Default is {'noun', 'verb', 'adjective'}

    Returns
    -------
    Embedding() object
        The word embedding object corresponding to the specified
        semantic embeddings
    """

    if vocab_limit == 0:
        addon = f'full_D{embedding_dim}'
    else:
        addon = f'V{vocab_limit}_D{embedding_dim}'

    # Path name of the returned objects
    matrix_name = f'def_matrix_{addon}.npz'
    idxdict_name = f'idx_words_to_matidx_{addon}.pkl'
    embed_name = f'def_embeddings_{addon}.npy'

    datafiles = os.listdir(Path('data'))
    if (idxdict_name in datafiles and
            embed_name in datafiles):
        print(str(datetime.now())+'\t'+'Importing semantic embeddings')

        # def_matrix = sparse.load_npz(Path('data') /
        #                              matrix_name.npz')
        idx_words_to_matidx = pickle.load(
            open(Path('data') / idxdict_name, 'rb'))
        def_embeddings = np.load(Path('data') / embed_name)

        Wiki_dict = pickle.load(open(Path('data') /
                                     'Wiktionary_dict.pkl', 'rb'))
        token_to_idx = {token: idx_words_to_matidx[idx_word]
                        for token, idx_word in Wiki_dict.items()
                        if idx_word in idx_words_to_matidx}

        return Embeddings(def_embeddings, token_to_idx)

    print(str(datetime.now())+'\t' +
          'No embedding files founds, computing embeddings')

    print(str(datetime.now())+'\t'+'Importing Wiktionary')

    W = wiktionary()

    print(str(datetime.now())+'\t'+'Creating vocabulary')
    time.sleep(0.5)
    idx_words = []

    for i, dat in tqdm(enumerate(W.data), total=len(W.data)):
        # word = dat.word

        # Counting occurences in synonyms
        if lang in dat.synonyms:
            for pos, syns in dat.synonyms[lang].items():
                token_count = []
                if pos.lower() in pos_list:
                    for syn in syns:
                        token = syn.replace('_', ' ')
                        if token in W.dict:
                            token_count.append(W.dict[token])
                idx_words += list(set(token_count))

        # Counting occurences in definitions
        if lang in dat.definitions:
            for pos, defins in dat.definitions[lang].items():
                if pos.lower() in pos_list:
                    idx_words.append(i)
                    for token in [token
                                  for defin in defins
                                  for token in tokenize(defin)
                                  if (token in W.dict and
                                      W.main_pos(token, lang=lang) in pos_list)
                                  ]:
                        idx_words.append(W.dict[token])

    counter_words = Counter(idx_words)
    counter_words_sort = counter_words.most_common()
    idx_words = [idx[0] for idx in counter_words_sort]

    print(str(datetime.now())+'\t'+f'{len(idx_words)} words found')

    if vocab_limit != 0:
        min_count = counter_words_sort[vocab_limit][1]
        while counter_words_sort[vocab_limit][1] <= min_count:
            vocab_limit -= 1
        min_count = counter_words_sort[vocab_limit][1]
        print(str(datetime.now()),
              f'Limiting size to {vocab_limit} most frequent words'
              f', the less common being: '
              f'"{W.data[idx_words[vocab_limit-1]].word}" '
              f'at {min_count} occurences in the corpus')

        idx_words = idx_words[:vocab_limit]
    idx_words_to_matidx = {i: mi for mi, i in enumerate(idx_words)}
    idx_words_set = set(idx_words)

    print(str(datetime.now())+'\t'+'Building linking matrix')
    time.sleep(0.5)
    # Idea: weighing the definitions by dependancy parsing rank?

    def_matrix = sparse.lil_matrix(sparse.eye(len(idx_words)),
                                   dtype='float64')

    for matidx, idx in tqdm(enumerate(idx_words), total=len(idx_words)):
        dat = W.data[idx]
        # word = dat.word
        # Putting synonyms in embeddings matrix
        if lang in dat.synonyms:
            # List of all valid synonyms token.
            # This should be it's own function cause it's cloned from
            # the previous section.
            idx_syns = []
            for pos, syns in dat.synonyms[lang].items():
                if pos.lower() in pos_list:
                    for syn in syns:
                        token = syn.replace('_', ' ')
                        if token in W.dict:
                            idx_syn = W.dict[token]
                            if idx_syn in idx_words_set and idx_syn != idx:
                                idx_syns.append(idx_syn)

            for matjdx in [idx_words_to_matidx[idx_syn]
                           for idx_syn in idx_syns]:
                def_matrix[matidx, matjdx] += 1/len(idx_syns)
                def_matrix[matjdx, matidx] += 1/len(idx_syns)

        if lang in dat.definitions:
            # List of all valid definitions token.
            # This should be it's own function cause it's cloned from
            # the previous section.
            all_idx_defs = []
            for pos, defins in dat.definitions[lang].items():
                if pos.lower() in pos_list:
                    idx_defs = []
                    for token in [token
                                  for defin in defins
                                  for token in tokenize(defin)
                                  if (token in W.dict and
                                      W.main_pos(token, lang=lang) in pos_list)
                                  ]:
                        idx_def = W.dict[token]
                        if idx_def in idx_words_set and idx_def != idx:
                            idx_defs.append(idx_def)
                    all_idx_defs.append(idx_defs)

            for idx_defs in all_idx_defs:
                for matjdx in [idx_words_to_matidx[idx_def]
                               for idx_def in idx_defs]:
                    def_matrix[matidx, matjdx] += \
                        1/len(idx_defs)/len(all_idx_defs)
                    def_matrix[matjdx, matidx] += \
                        1/len(idx_defs)/len(all_idx_defs)

    # Deleting Wiktionary data from memory
    token_to_idx = {token: idx_words_to_matidx[idx_word]
                    for token, idx_word in W.dict.items()
                    if idx_word in idx_words_to_matidx}
    del W
    def_matrix = sparse.csc_matrix(def_matrix)

    sparse.save_npz(Path('data') / matrix_name,
                    def_matrix)
    pickle.dump(idx_words_to_matidx,
                open(Path('data') / idxdict_name, 'wb'))

    print(str(datetime.now())+'\t'+'Normalization')
    time.sleep(0.5)

    def_norms = np.array(def_matrix.sum(axis=0)).reshape(-1)
    xcor, ycor, vals = sparse.find(def_matrix)
    norm_def_matrix = def_matrix.tolil()
    for i in tqdm(range(vals.shape[0])):
        norm_def_matrix[xcor[i], ycor[i]] /= \
            def_norms[xcor[i]]*def_norms[ycor[i]]

    print(str(datetime.now())+'\t'+'Diagonalisation')

    eig_val, eig_vec = eigsh(norm_def_matrix, k=embedding_dim)

    print(str(datetime.now())+'\t'+'Saving')

    def_embeddings = def_matrix.dot(eig_vec)
    np.save(Path('data') / embed_name,
            def_embeddings)

    return Embeddings(def_embeddings, token_to_idx)


class Embeddings():
    """Word embedding object

    Parameters
    ----------
    emb_matrix: Numpy 2D array
        Word embedding matrix. Each column corresponds to a word embedding
        vector of a specified dimension

    token_to_idx: dict of str:int
        dictionary where the key is a word token string, and the
        index is the integer index of the column in emb_matrix
        corresponding to the proper embedding

    name: str, optional
        Name of the embedding. Default is None.
    """

    def __init__(self,
                 emb_matrix,
                 token_to_idx,
                 name=None):
        self.emb_matrix = emb_matrix
        self.token_to_idx = token_to_idx
        self.name = name

    def emb(self,
            token):
        """Returns the embedding vector corresponding to the
        specified word token.

        If the token is not in vocabulary, returns None.

        Parameters
        ----------
        token: str
            The specified word token
        """
        if token in self.token_to_idx:
            return self.emb_matrix[self.token_to_idx[token]]
        else:
            return None

    def sim_testing(self,
                    all_pairs):
        """Returns spearman correlation of pair of words similarity ordering
        from the specified dataset pairs.

        Prints statistics in console.

        Parameters
        ----------
        all_pairs: dict of str:list of tuples
            Dict object where key=(str) the name of the dataset and
            value=similarity data of the dataset
            Each similarity data those is a list of at least 3-tuple
            (str, str, float) which are
            (Word1, Word2, similarity between words between 0 and 1)
            Some datasets can have longer tuples with part-of-speech
            data but additional data will be ignored here.
            This object is returned by import_benchmark() function
        """
        max_len_name = max([len(x) for x in all_pairs])
        for dataset_name, pairs in all_pairs.items():
            data_embed = []
            data_bench = []
            for pair in pairs:
                word1 = self.emb(pair[0])
                word2 = self.emb(pair[1])
                bench_sim = pair[2]
                if word1 is not None and word2 is not None:
                    embed_sim = (np.dot(word1, word2) /
                                 (np.linalg.norm(word1)*np.linalg.norm(word2)))
                    data_embed.append(embed_sim)
                    data_bench.append(bench_sim)
            correlation = spearmanr(data_embed, data_bench)[0]
            set_name = (dataset_name +
                        ' '*max(0, max_len_name+2-len(dataset_name)))
            word_id = 100*len(data_bench)/len(pairs)

            print(f'{set_name}'
                  f'Spearman: {correlation:.2f}   '
                  f'Word_id: {word_id:4.1f}% '
                  f'({len(data_bench):4} / {len(pairs):4})')

    def closest_words(self,
                      token,
                      n=20):
        """Prints the top {n} closest word tokens to the specified word

        Parameters
        ----------
        token: str
            The specified word token

        n: int, optional
            The number of closest neighbors to print out. Default is 20.
        """
        vector = self.emb(token)
        if vector is None:
            print(f'{token} is not in dictionary')
            return
        print(f'\tClosest neighbours of "{token}"')
        idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        vector = vector / np.linalg.norm(vector)
        norm_embeddings = (self.emb_matrix.T /
                           np.linalg.norm(self.emb_matrix, axis=1)).T
        distances = np.dot(norm_embeddings, vector)
        idxs = np.argsort(distances)[-n:][::-1]

        print_data = []
        n = 0
        max_len = 10
        for idx in idxs:
            if idx != self.token_to_idx[token]:
                word = idx_to_token[idx]
                max_len = max(max_len, len(word))
                print_data.append([word, distances[idx]])
            n += 1
            if n >= 20:
                break
        print('Word_token' + ' '*(max_len-10) + '  cosine_sim')
        for word, distance in print_data:
            set_word = word + ' '*(max_len-len(word))
            print(f'{set_word}  {distance:.3f}')

    def plot_synstats(self,
                      syn_pairs):
        """Plots the ranks of a word's synonym when ordered
        by closest embedding neighbour for all synonyms in Wiktionary

        Parameters
        ----------
        syn_pairs: list of (str, str)
            List of pair of word token that are synonyms
        """
        print(str(datetime.now())+'\t'+'Generating synonym rank plot')

        norm_embeddings = (self.emb_matrix.T /
                           np.linalg.norm(self.emb_matrix, axis=1)).T

        widx_pairs = [[self.token_to_idx[w1], self.token_to_idx[w2]]
                      for (w1, w2) in syn_pairs
                      if w1 in self.token_to_idx and w2 in self.token_to_idx]

        syn_stats = []
        size_batches = 100
        batches = [[i*size_batches, (i+1)*size_batches]
                   for i in range((len(widx_pairs)-1)//size_batches+1)]
        for i1, i2 in tqdm(batches):
            widxes = np.array(sorted(list(set(
                [widx for widx_pair in widx_pairs[i1:i2]
                 for widx in widx_pair]
                ))))
            rank_matrix = (norm_embeddings@norm_embeddings[widxes].T).T
            rank_matrix = np.flip(rank_matrix.argsort(axis=1), axis=1)
            rank_matrix = rank_matrix.argsort(axis=1)
            for widx1, widx2 in widx_pairs[i1:i2]:
                syn_stats.append(
                    rank_matrix[np.searchsorted(widxes, widx1)][widx2]
                    )
                syn_stats.append(
                    rank_matrix[np.searchsorted(widxes, widx2)][widx1]
                    )

        plot_data = list(map(list, zip(*Counter(syn_stats).items())))
        # Normalization the number of pairs to 100000 for comparison
        # with other embeddings
        C = 100000/len(syn_stats)
        y_data = [y*C for y in plot_data[1]]

        plt.loglog(plot_data[0],
                   y_data,
                   '.',
                   label=None)
        plt.title('Y: count number, X: Rank of a word\'s synonym '
                  'when ordered by closest embedding neighbour')
        plt.legend()
        plt.show()
        print(f'Avg Rank {np.mean(syn_stats):.1f}, '
              f'Median Rank {np.median(syn_stats)} '
              f'(on {len(syn_stats)} pairs)')


def import_synpairs(lang='english',
                    pos_list={'noun', 'verb', 'adjective'}):
    """Imports synonym pairs from Wiktionary data.

    Uses data file ./data/synpairs.pkl
    If the file is absent, it will generate it.

    It will only grab paris that are single token words
    and whose unidecode translation differ.

    Parameters
    ----------
    lang: str, optional
        Language to consider for the embedding. Must correspond to a
        key in the Definition() object from the Wiktionary input file
        from wiktionary() function.
        Default is 'english'

    pos_list: dict of str, optional
        Parts-of-speech to consider for the word tokens. Every other
        part-of-speech will be completely ignored. Must correspond to a
        key in the Definition() object from the Wiktionary input file
        from wiktionary() function.
        Default is {'noun', 'verb', 'adjective'}

    Returns
    -------
    list of (str, str)
        list of word token pairs that are synonyms
    """

    filename = 'synpairs.pkl'
    if filename in os.listdir(Path('data')):
        print(str(datetime.now())+'\t'+'Importing synonym pairs')
        synpairs = pickle.load(open(Path('data') / filename, 'rb'))
        return synpairs

    print(str(datetime.now())+'\t'+'Generating synonym pairs')

    W = wiktionary()
    syn_pairs = []
    for dat in W.data:
        word = dat.word
        if lang in dat.synonyms:
            # List of all valid synonyms token.
            # This should be it's own function cause it's cloned from
            # a previous section.
            for pos, syns in dat.synonyms[lang].items():
                if pos.lower() in pos_list:
                    for syn in syns:
                        token = syn.replace('_', ' ')
                        syn_pairs.append((word, token))
    syn_pairs = sorted(list(set(syn_pairs)))

    syn_pairs = [(w1, w2) for (w1, w2) in syn_pairs
                 if (len(w1.replace('_', '').split()) == 1 and
                     len(w2.replace('_', '').split()) == 1 and
                     unidecode(w1).lower() != unidecode(w2).lower())]
    pickle.dump(syn_pairs,
                open(Path('data') / filename, 'wb'))

    return syn_pairs


if __name__ == '__main__':
    all_pairs = import_benchmark()
    D = import_semantic(vocab_limit=10000)
    print('\n\tSemantic (vocab=10K) embedding similarity statistics')
    D.sim_testing(all_pairs)
    print()
    D.closest_words('old')
    syn_pairs = import_synpairs()
    D.plot_synstats(syn_pairs)
