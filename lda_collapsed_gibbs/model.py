import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from pprint import pprint

module_path = '/Users/ryanho/Documents/python/focal/text_clustering/find_keyword/lda_collapsed_gibbs/'
sys.path.append(module_path)
from sampler import Sampler


"""
watch every sampling result from lda sampler
record the sampling performance
update the best sampling
get the optimal distribution 1. phi = p(w| z)  2. pzw = p(z| w)  in 'maxiter' times sampling
and save them to local if necessary 

the optimal distribution will then be used in word_predictor   

"""


class LdaModel:

    def __init__(self, config):
        self.config = config
        self.maxiter = config['maxiter']

        self.alpha = None
        self.beta = None
        self.n_topics = None

        self.opt_phi = None    # optimal p(w| z)
        self.opt_pzw = None   # optimal p(z| w)

        self.opt_iter = None   # optimal iteration
        self.maxlike = None    # best maximum likelihood

        self.likelihood_in_iters = None

    def build(self, td_matrix, alpha, beta, n_topics, save_model=False):
        """
        run Sampler and update p(z|w) and p(w|z)

        """

        print('build model')

        self._initialize()

        sampler = Sampler(n_topics=n_topics, alpha=alpha, beta=beta)

        # extract 1. phi   2. nmz   3. nzw from sampler every iteration(sampling)
        # phi_nmz_nzw = (phi , nmz, nzw)  generator return a tuple for multiple values
        for i, phi_pzw in enumerate(sampler.run(matrix=td_matrix, maxiter=self.maxiter)):
            like = sampler.loglikelihood()

            self.likelihood_in_iters[i] = like

            # update best maximum likelihood and optimal phi = p(w| z)
            if like > self.maxlike:
                self.maxlike = like
                self.opt_iter = i
                self.opt_phi = phi_pzw[0]
                self.opt_pzw = phi_pzw[1]

        if save_model is True:
            self._save_lda_model()

    def get_p_zw(self):
        """
        the topic proportion of words

        Also, the feature vector of words

        p_zw.shape(w, z)

        """

        return self.opt_pzw   # p_zw.shape(w, z)

    def get_p_wz(self):
        """
        the word distribution of topics

        phi.shape (z, w)
        p_wz.shape (z, w)

        """
        return self.opt_phi

    def get_maxlike(self):
        return self.maxlike

    def show_optimal(self):
        print('---- config ----')
        print('alpha : '+str(self.alpha))
        print('beta : '+str(self.beta))
        print('n_topics : '+str(self.n_topics))
        print('---- optimal ----')
        print('maximum likelihood : '+str(self.maxlike))
        print('opt_iter '+str(self.opt_iter))

    def show_maxlike_changes(self):
        iter = list(self.likelihood_in_iters.keys())
        likelihood = list(self.likelihood_in_iters.values())

        plt.plot(iter, likelihood)
        plt.xlabel('iteration')
        plt.ylabel('loglikelihood')
        plt.show()

    def show_word_topic_distribution(self, word):
        p_zw = np.load(self.config['path'] + 'p_zw' + self.config['model_ver'] + '.npy')

        with open(self.config['path'] + 'word_id_converter.json', 'r', encoding='utf8') as f:
            word_id_converter = json.load(f)

        word2id = word_id_converter['word2id']

        word_distribution = dict()
        if type(word) is list:
            for w in word:
                if w in word2id.keys():
                    wordid = word2id[w]
                    distribution = p_zw[wordid, :]
                    word_distribution[w] = distribution

                else:
                    print(str(w)+' not in dictionary')

        elif type(word) is str:
            if word in word2id.keys():
                wordid = word2id[word]
                distribution = p_zw[wordid, :]
                word_distribution[word] = distribution

            else:
                print(str(word) + ' not in dictionary')

        pprint(word_distribution)

    def _initialize(self):
        self.maxlike = -1*10**100
        self.opt_iter = 0

        self.opt_phi = np.arange(1)

        self.likelihood_in_iters = dict()

    def _save_lda_model(self):
        print('save lda model')
        model_info = dict()
        model_info['maximum_likelihood'] = self.maxlike  # int
        model_info['optimal_iteration'] = self.opt_iter
        model_info['likelihood_in_iters'] = self.likelihood_in_iters

        with open(self.config['path'] + 'lda_model_info' + self.config['model_ver'] + '.json', 'w', encoding='utf8') as f:
            json.dump(model_info, f)

        np.save(self.config['path'] + 'p_wz' + self.config['model_ver'] + '.npy', self.opt_phi)
        np.save(self.config['path'] + 'p_zw' + self.config['model_ver'] + '.npy', self.opt_pzw)
