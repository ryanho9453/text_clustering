"""
term to term similarity matrix

use terms' feature to calcualte cosine similarity between words

tt_similarity.shape(n_words, n_words)

similarity between john, mary = tt_similarity[johnid, maryid] = tt_similarity[maryid, johnid]

"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TermSimilarityMatrix:
    def __init__(self, config):
        self.config = config

    def create(self, term_feature_maxtrix, save=False):
        term_similarity_matrix = cosine_similarity(term_feature_maxtrix, term_feature_maxtrix)

        if save is True:
            np.save(self.config['path'] + 'term_similarity_matrix' + self.config['model_ver'] + '.npy', term_similarity_matrix)

        return term_similarity_matrix
