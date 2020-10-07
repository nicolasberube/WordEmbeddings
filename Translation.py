#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:05:40 2020

@author: berube
This code uses a pre-trained map of multilingual embeddings (e.g. Glove) to
translate another embedding space (e.g. Paragrams)

It does so by projecting a vector in the target language (e.g. French) in the
orthonormal basis spanned by the (ordered) Gram-Schmidt decomposition of the
K (<= n_dim) nearest neighbours in the source language (e.g. English)

It then creates another orthonormal basis spanned by the (ordered)
Gram-Schmidt decomposition of the same tokens in the source language
e.g. English), but in the new embeddings space

With this new basis, it recreates the target language (e.g. French)
vector in the new embedding space

note: the K nearest neighbours are measured based on CSLS
(Cross-Lingual Similarity Scaling) distance from
"Word translation without parallel data", Conneau et al., ICLR 2018

This process supposes two things:

a) the K (<= n_dim) nearest neighbours in multilingual embeddings space is
   enough to construct an accurate translation of the token

b) The cosine distance between vectors from different languages in
   multilingual embeddings is preserved between embedding space
   (e.g. between Glove and Paragrams)

In other words, the relationships between "cat" and "tiger" can differ
between embedding spaces (Glove might put them far because they are used in
different contexts, but Paragrams might put them close cause they are both
felines)

but what matters is that the relationships between "chat" and "cat"
(and "tigre" and "tiger") is the same in both multilingual space
(both Glove and Paragrams would put them very close cause they are
translations of each others)

The limit of this supposition is complex translations (e.g. "feuille" is
gonna be close to both "sheet" and "leaf", but the relationship between
"sheet" and "leaf" will vary between embedding spaces, so this will not
respect axiom b)
"""

import numpy as np
from pathlib import Path
import time
from datetime import datetime
from numpy.linalg import norm
import os
# from shutil import copyfile
# import logging
# import sys
import pickle
import torch
from tqdm import tqdm
from Semantic import Embeddings


def translate(emb_orig,
              multi_orig,
              multi_dest,
              benchmark=False):
    """Translates word embeddings from an origin language to a destination
    language, based on anther multilingual embeddings mapping those languages
    to the same latent space.

    It will save the embeddings in ./data/ folder in three files
    - a Numpy file of the embeddings data where each column
      is a word's embedding vector
    - a Pickle file contaning a dictionary where key=word token (str)
      and value=idx of the embedding matrix corresponding to the word's data
    - a txt file containing the above information

    It will also save a temporary Numpy file "rdist" to help with calculation

    The names of the file will depend on the Embeddings() name property
    in the input.

    Parameters
    ----------
    emb_orig: Embeddings() class
        The embessings to translate, in the origin language

    multi_orig: Embeddings() class
        The multilingual embeddings in the origin language

    multi_dest: Embeddings() class
        The multilingual embeddings in the destination language

    benchmark: bool, optional
        Will run a check on the embeddings after their translation
        to verify if the axioms of the translation process
        are respected
    """

    # Calculating CSLS distance matrices
    print(str(datetime.now())+'\t'+'Calculating CSLS distances for'
          'starting language (compared to the second language)')

    start_time = time.time()

    print(str(datetime.now())+'\t'+'Creating rdist file on disk and '
          'monitoring progress from last calculation')

    # Path of the rdist matrix which contains reuseable CSLS distance
    # calculation
    name_multi = multi_orig.name
    if name_multi is None:
        name_multi = 'multi'

    name_orig = emb_orig.name
    if name_orig is None:
        name_orig = 'orig'
    # Path for the temporary rdist file
    path_rdist = Path('data') / f'rdist_{name_multi}_{name_orig}.npy'

    # Path for the numpy matrix of the new translated embeddings
    path_emb_dest_mat = Path('data') / f'{name_orig}_emb.npy'
    # Same object but in txt format
    path_emb_dest_txt = Path('data') / f'{name_orig}_emb.txt'
    # Path for the token_to_idx object of the new translated embeddings
    path_emb_dest_tti = Path('data') / f'{name_orig}_tti.pkl'

    # Vocabulary limitation to intersection of multilingual embeddings
    # multi_orig and embeddings to translate emb_orig
    vocab_orig = set(emb_orig.token_to_idx)
    vocab_orig = list(vocab_orig.intersection(set(multi_orig.token_to_idx)))

    multi_arrayidx = [multi_orig.token_to_idx[t] for t in vocab_orig]
    emb_arrayidx = [emb_orig.token_to_idx[t] for t in vocab_orig]
    vocab_orig = {t: i for i, t in enumerate(vocab_orig)}

    multi_orig = Embeddings(multi_orig.emb_matrix[multi_arrayidx],
                            vocab_orig)
    emb_orig = Embeddings(emb_orig.emb_matrix[emb_arrayidx],
                          vocab_orig)

    # Calculating memory map parameters for conversion between
    # numpy and memmaps

    k_max = multi_orig.emb_matrix.shape[1]
    size = (multi_orig.emb_matrix.shape[0], k_max)

    # R-distances will be saved in 8-byte floats.
    # typesize is the number of byte of the r-distance matrix
    type_size = 8

    # Saving the matrix as a full numpy object with header.
    # This step is so that the object can be imported with numpy later
    # even if it's built with memmaps.
    if not os.path.isfile(path_rdist):
        rdist = np.zeros(size)
        np.save(path_rdist, rdist)
        del rdist

    # Loading the memmap
    offset = os.path.getsize(path_rdist)-size[0]*size[1]*type_size
    rdist = np.memmap(path_rdist,
                      dtype=('f%i' % type_size),
                      mode='r+',
                      offset=offset,
                      shape=size)

    # Checking what part of the matrix has already been computed
    idx_start = 0
    while idx_start < rdist.shape[0]:
        if rdist[idx_start].all() == 0:
            break
        else:
            idx_start += 1

    if idx_start == rdist.shape[0]:
        print(str(datetime.now()) + '\t' +
              'ridst matrix already found')

    else:
        print(str(datetime.now()) + '\t' +
              f'ridst matrix found up to idx_start = {idx_start}')

        print(str(datetime.now())+'\t'+'Calculating rdist')

        # Normalizing multilingual destination language vectors
        gpu_flag = torch.cuda.is_available()
        if gpu_flag:
            multi_dest_norm = torch.cuda.DoubleTensor(multi_dest.emb_matrix)
        else:
            multi_dest_norm = torch.DoubleTensor(multi_dest.emb_matrix)
        multi_dest_norm = torch.nn.functional.normalize(multi_dest_norm)

        # Batch of origin language vectors to calculate r
        # for future CSLS calculation
        # r_ik = mean_j=1_to_k_nearest_neighbours(cos_distance(x_i, y_j))
        # where x_i is a vector in the destination language and y_j are
        # the sorted closest vectors in the origin language
        # Will be done on GPU if possible
        batch_size = 100
        total_n = size[0]
        batch_num = (total_n-1)//batch_size+1
        start_flag = False

        time.sleep(0.5)
        for i in tqdm(range(batch_num)):
            idx1 = i*batch_size
            idx2 = min((i+1)*batch_size, total_n)

            if idx2 >= idx_start and idx1 <= idx_start:
                start_flag = True
            if start_flag:
                if gpu_flag:
                    emb_orig_norm = torch.cuda.DoubleTensor(
                        multi_orig.emb_matrix[idx1:idx2]
                        )
                else:
                    emb_orig_norm = torch.DoubleTensor(
                        multi_orig.emb_matrix[idx1:idx2]
                        )
                emb_orig_norm = torch.nn.functional.normalize(emb_orig_norm)
                dists_cos = 1 - torch.mm(multi_dest_norm,
                                         emb_orig_norm.t()).t()
                sorted_idx = dists_cos.sort()[0]
                for k in range(k_max):
                    if gpu_flag:
                        rdist[idx1:idx2, k] = sorted_idx[:, :k+1].mean(1).cpu()
                    else:
                        rdist[idx1:idx2, k] = sorted_idx[:, :k+1].mean(1)

        # Freeing memory
        del emb_orig_norm
        del multi_dest_norm

        time.sleep(0.5)
        print(f'{(time.time()-start_time):.1f} sec')

    # Translation

    print(str(datetime.now())+'\t'+'Creating destination language file on'
          'disk and monitoring progress from last calculation')

    # k_nn is the number of nearest neighbours to be used in the CSLS distance
    # calculation
    k_nn = emb_orig.emb_matrix.shape[1]
    size = (multi_dest.emb_matrix.shape[0], k_nn)

    # Embeddings will be saved in 8-byte floats.
    # typesize is the number of bytes.
    type_size = 8

    # Saving the matrix as a full numpy object with header.
    # This step is so that the object can be imported with numpy later
    # even if it's built with memmaps.
    if not os.path.isfile(path_emb_dest_mat):
        emb_dest_mat = np.zeros(size)
        np.save(path_emb_dest_mat, emb_dest_mat)
        del emb_dest_mat

    offset = os.path.getsize(path_emb_dest_mat)-size[0]*size[1]*type_size
    emb_dest_mat = np.memmap(path_emb_dest_mat,
                             dtype=('f%i' % type_size),
                             mode='r+',
                             offset=offset,
                             shape=size)

    idx_start = 0
    while idx_start < emb_dest_mat.shape[0]:
        if emb_dest_mat[idx_start].all() == 0:
            break
        else:
            idx_start += 1

    if idx_start == emb_dest_mat.shape[0]:
        print(str(datetime.now()) + '\t' +
              'Translated embedding matrix already present')
    else:
        print(str(datetime.now()) + '\t' +
              f'Translated embedding matrix found up to '
              f'idx_start = {idx_start}')

        start_time = time.time()
        # The following code reuses part of the previous onde since it's
        # the other half of the CSLS distance
        # They maybe could get mixed into a single method

        # Normalizing multilingual origin language vectors
        if gpu_flag:
            multi_orig_norm = torch.cuda.DoubleTensor(multi_orig.emb_matrix)
            rdist_k = torch.cuda.DoubleTensor(rdist.T[k_nn-1])
        else:
            multi_orig_norm = torch.DoubleTensor(multi_orig.emb_matrix)
            rdist_k = torch.DoubleTensor(rdist.T[k_nn-1])
        multi_orig_norm = torch.nn.functional.normalize(multi_orig_norm)

        # Batch of destination language embedding vectors to calculate
        batch_size = 100
        batch_num = (multi_dest.emb_matrix.shape[0]-1)//batch_size+1
        start_flag = False

        time.sleep(0.5)
        for i in tqdm(range(batch_num)):
            idx1 = i*batch_size
            idx2 = min((i+1)*batch_size, multi_dest.emb_matrix.shape[0])

            if idx2 >= idx_start and idx1 <= idx_start:
                start_flag = True
            if start_flag:
                # This calculation was done earlier in the calculation of rdist
                # but it is remade here to save memory on the GPU if needed be
                if gpu_flag:
                    multi_dest_norm = torch.cuda.DoubleTensor(
                        multi_dest.emb_matrix[idx1:idx2]
                        )
                else:
                    multi_dest_norm = torch.DoubleTensor(
                        multi_dest.emb_matrix[idx1:idx2]
                        )
                multi_dest_norm = \
                    torch.nn.functional.normalize(multi_dest_norm)

                # Finding K nearest neighbours based on CSLS distance
                dists_cos = 1 - torch.mm(multi_orig_norm,
                                         multi_dest_norm.t()).t()
                sorted_idx = dists_cos.sort()[0]
                # The CSLS distance
                dists_cos = ((2*dists_cos-rdist_k).t() -
                             sorted_idx[:, :k_nn].mean(1)).t()
                # Identifying the closest vectors based on CSLS distance
                sorted_idx = dists_cos.sort()[1]

                # CPU calculation for Gram-Schmidt
                for idx in range(idx2-idx1):
                    # Creating orthonormal basis in multilingual space
                    precision = 10**-10
                    # Gram-Schmidt basis vector matrix
                    gs_basis = np.zeros((k_nn, multi_orig.emb_matrix.shape[1]))
                    # Sorted index
                    so_idx = 0
                    # Gram Schmidt index, which might be desynced from the
                    # so_idx if the next vector is a linear combination of
                    # all previous ones
                    gs_idx = 0
                    while gs_idx < k_nn:
                        new_vect = \
                            multi_orig.emb_matrix[sorted_idx[idx][so_idx]]
                        gs_vect = (new_vect -
                                   np.dot(np.dot(new_vect,
                                                 gs_basis[:gs_idx].T),
                                          gs_basis[:gs_idx]))
                        norm_gsv = norm(gs_vect)
                        if norm_gsv > precision:
                            gs_basis[gs_idx] = gs_vect/norm_gsv
                            gs_idx += 1
                        so_idx += 1

                    # Creating orthonormal basis in new space
                    # with the same tokens
                    new_gs_basis = np.zeros((k_nn,
                                             emb_orig.emb_matrix.shape[1]))
                    so_idx = 0
                    gs_idx = 0
                    while gs_idx < k_nn:
                        new_vect = emb_orig.emb_matrix[sorted_idx[idx][so_idx]]
                        gs_vect = (new_vect -
                                   np.dot(np.dot(new_vect,
                                                 new_gs_basis[:gs_idx].T),
                                          new_gs_basis[:gs_idx]))
                        norm_gsv = norm(gs_vect)
                        if norm_gsv > precision:
                            new_gs_basis[gs_idx] = gs_vect/norm_gsv
                            gs_idx += 1
                        so_idx += 1

                # New embedding
                    emb_dest_mat[idx1+idx] = \
                        np.dot(np.dot(multi_dest_norm[idx],
                                      gs_basis.T),
                               new_gs_basis)

        time.sleep(0.5)
        print(f'{(time.time()-start_time):.1f} sec')

        # Saving embeddings
        print(str(datetime.now())+'\t'+'Saving embeddings')
        pickle.dump(multi_dest.token_to_idx, open(path_emb_dest_tti, 'wb'))

        # Saving in txt format
        with open(path_emb_dest_txt, 'w', encoding='utf-8') as fo:
            for token, idx in multi_dest.token_to_idx.items():
                if idx != 0:
                    fo.write('\n')
                if ' ' not in token:
                    fo.write(token+' '+' '.join([str(x)
                                                 for x in emb_dest_mat[idx]]))
        print(str(datetime.now())+'\t'+'Saving done')

    if not benchmark:
        return
    print(str(datetime.now())+'\t'+'Testing new embeddings')
    # Calculating if the axioms a) and b) are true
    # a) the K (<= n_dim) nearest neighbours in multilingual embeddings space
    #    is enough to construct an accurate translation of the token
    # b) The cosine distance between vectors from different languages in
    #    multilingual embeddings is preserved between embedding space

    # Data will be saved in numpy matrix at the following path
    # The values on this matrix is per embedding, and it should be zero
    name_orig = emb_orig.name
    if name_orig is None:
        name_orig = 'orig'
    path_loss = Path('data') / f'{name_orig}_loss.npy'

    n_tokens = emb_dest_mat.shape[0]

    # For real, this following code to make sure numpy object is properly
    # through memmaps is used all throughout this library and should be
    # it's own function or class

    # Embeddings are saved with 8-byte floats.
    # typesize is the number of bytes
    type_size = 8
    if not os.path.isfile(path_loss):
        avg_loss = avg_loss = np.zeros((n_tokens, 2))
        np.save(path_loss, avg_loss)
        del avg_loss

    offset = os.path.getsize(path_loss) - n_tokens*2*type_size
    avg_loss = np.memmap(path_loss,
                         dtype=('f%i' % type_size),
                         mode='r+',
                         offset=offset,
                         shape=(n_tokens, 2))
    for idx_start in range(n_tokens):
        if avg_loss[idx_start].all() == 0:
            break
    idx_start -= 1
    idx_start = max(idx_start, 0)

    print(str(datetime.now()) + '\t' +
          f'loss matrix found up to idx_start = {idx_start}')

    time.sleep(0.5)
    for idx in tqdm(range(n_tokens)):
        if idx >= idx_start:
            dists_cos_all = 1 - (np.dot(multi_orig.emb_matrix,
                                        multi_dest.emb_matrix[idx]) /
                                 norm(multi_orig.emb_matrix, axis=1) /
                                 norm(multi_dest.emb_matrix[idx]))
            dists_euc_all = norm(multi_orig.emb_matrix -
                                 multi_dest.emb_matrix[idx], axis=1)

            new_dists_cos_all = 1 - (np.dot(emb_orig.emb_matrix,
                                            emb_dest_mat[idx]) /
                                     norm(emb_orig.emb_matrix, axis=1) /
                                     norm(emb_dest_mat[idx]))
            new_dists_euc_all = norm(emb_orig.emb_matrix -
                                     emb_dest_mat[idx], axis=1)

            avg_loss_cos = abs(dists_cos_all - new_dists_cos_all).mean()
            avg_loss_euc = abs(dists_euc_all - new_dists_euc_all).mean()
            avg_loss[idx] = [avg_loss_cos, avg_loss_euc]
    time.sleep(0.5)
    print(str(datetime.now())+'\t'+'End')


if __name__ == '__main__':

    from Semantic import import_semantic
    # Import test data
    D = import_semantic(vocab_limit=10000)

    # Multilingual embeddings, from origin language
    multi_orig = Embeddings(D.emb_matrix[500:2000],
                            {t: i-500 for t, i in D.token_to_idx.items()
                             if 500 <= i < 2000})

    # Multilingual embeddings, for destination language
    multi_dest = Embeddings(D.emb_matrix[2000:4000],
                            {t: i-2000 for t, i in D.token_to_idx.items()
                             if 2000 <= i < 4000})

    # Embeddings to translate, in origin language
    emb_orig = Embeddings(np.random.random((1000, 300)),
                          {t: i-1000 for t, i in D.token_to_idx.items()
                           if 1000 <= i < 2000})

    translate(emb_orig, multi_orig, multi_dest, benchmark=True)

    emb_dest = Embeddings(
        np.load(Path('data') / 'orig_emb.npy'),
        pickle.load(open(Path('data') / 'orig_tti.pkl', 'rb'))
        )
