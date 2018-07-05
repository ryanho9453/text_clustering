"""
(C) Mathieu Blondel - 2010
License: BSD 3 clause

Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in

Finding scientifc topics (Griffiths and Steyvers)
"""

import sys
import numpy as np
from scipy.special import gammaln
import json

module_path = '/Users/ryanho/Documents/python/focal/text_clustering/find_keyword/tools/'
sys.path.append(module_path)
from tools import Timer


with open('/Users/ryanho/Documents/python/focal/text_clustering/find_keyword/config.json', 'r', encoding='utf8') as f:
    config = json.load(f)


def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """

    A = np.random.multinomial(1, p).argmax()

    return A


def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """

    # 若doc中有 a個Wa b個Wb ,則傳回Wa a次, 傳回Wb b次

    # idx 是td_matrix中 word的index, 只是這只取doc 中 nonzero 的字 )
    for idx in vec.nonzero()[0]:        # 取出doc 包含的word的id
        for i in range(int(vec[idx])):  # 若doc 中有n個wi, 傳回n 個 wi
            yield idx


def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)


class Sampler:

    def __init__(self, n_topics, alpha=0.1, beta=0.1):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

        self.prior_wordids, self.wordid2cluster = self._prepare_prior_words()

    def _prepare_prior_words(self):
        with open(config['path'] + 'word_id_converter.json', 'r', encoding='utf8') as f:
            word_id_converter = json.load(f)

        word2id = word_id_converter['word2id']

        with open(config['package_path'] + 'lda_collapsed_gibbs/word_cluster.json', 'r', encoding='utf8') as f:
            word_cluster = json.load(f)

        prior_wordids = list()
        wordid2cluster = dict()
        for word, cluster in word_cluster.items():
            if word in word2id.keys():
                wordid = word2id[word]
                prior_wordids.append(wordid)
                wordid2cluster[wordid] = cluster

        return prior_wordids, wordid2cluster

    def _initialize(self, matrix):
        n_docs, vocab_size = matrix.shape

        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics))
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, vocab_size))

        # n_words in docs
        self.nm = np.zeros(n_docs)

        # n_words in topics
        self.nz = np.zeros(self.n_topics)
        self.topics = {}

        for m in range(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                self.nmz[m, z] += 1
                self.nm[m] += 1
                self.nzw[z, w] += 1
                self.nz[z] += 1
                self.topics[(m, i)] = z

    def _conditional_distribution(self, m, w):
        """
        Conditional distribution (vector of size n_topics).
        """

        if w in self.prior_wordids:
            if 'in' in self.wordid2cluster[w].keys():
                p_z_with_prior = np.zeros((1, self.n_topics))
                prior_cluster = self.wordid2cluster[w]['in'][0]

                p_z_with_prior[:, prior_cluster] = 0.8

                other_index = [i for i in range(self.n_topics)]
                other_index.remove(prior_cluster)

                p_z_with_prior[:, other_index] = 0.2 / len(other_index)

                p_z_with_prior /= np.sum(p_z_with_prior)

                p_z_with_prior = p_z_with_prior.tolist()[0]

                return p_z_with_prior

            elif 'not_in' in self.wordid2cluster[w].keys():
                vocab_size = self.nzw.shape[1]
                left = (self.nzw[:, w] + self.beta) / \
                       (self.nz + self.beta * vocab_size)
                right = (self.nmz[m, :] + self.alpha) / \
                        (self.nm[m] + self.alpha * self.n_topics)
                p_z = left * right
                # normalize to obtain probabilities
                p_z /= np.sum(p_z)

                excluding_clusters = self.wordid2cluster[w]['not_in']   # list of clusters

                for cluster in excluding_clusters:
                    p_z[cluster] = 0

                p_z /= np.sum(p_z)

                return p_z

            else:
                print('no in or not_in in word_cluster.json for the word')
                sys.exit(1)

        else:
            vocab_size = self.nzw.shape[1]
            left = (self.nzw[:, w] + self.beta) / \
                (self.nz + self.beta * vocab_size)
            right = (self.nmz[m, :] + self.alpha) / \
                (self.nm[m] + self.alpha * self.n_topics)
            p_z = left * right

            # normalize to obtain probabilities
            p_z /= np.sum(p_z)

            return p_z

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """

        vocab_size = self.nzw.shape[1]
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in range(self.n_topics):
            lik += log_multi_beta(self.nzw[z, :]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in range(n_docs):
            lik += log_multi_beta(self.nmz[m, :]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def phi_pzw(self):
        """
        Compute phi = p(w|z).
        """

        phi = self.nzw + self.beta
        phi /= np.sum(phi, axis=1)[:, np.newaxis]

        p_zw = self.nzw.copy()
        p_zw /= np.sum(p_zw, axis=0)[np.newaxis, :]

        return [phi, p_zw.T]   # phi.shape (z, w)  pzw.shape(w, z)

    def run(self, matrix, maxiter=30):   # matrix shape(#doc, #word)
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = matrix.shape

        self._initialize(matrix)

        for it in range(maxiter):
            timer = Timer()
            timer.start()

            print('--- iter '+str(it))

            for m in range(n_docs):
                # 在doc m下的第i個字 --  w (此處的w是td_matrix中，word的index)
                for i, w in enumerate(word_indices(matrix[m, :])):
                    z = self.topics[(m, i)]
                    self.nmz[m, z] -= 1
                    self.nm[m] -= 1
                    self.nzw[z, w] -= 1
                    self.nz[z] -= 1

                    p_z = self._conditional_distribution(m, w)
                    z = sample_index(p_z)

                    self.nmz[m, z] += 1
                    self.nm[m] += 1
                    self.nzw[z, w] += 1
                    self.nz[z] += 1
                    self.topics[(m, i)] = z

            timer.print_time()
            print('--- end iter')

            # FIXME: burn-in and lag!
            yield self.phi_pzw()
