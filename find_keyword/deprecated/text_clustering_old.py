import numpy as np
import gc
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
import faulthandler
faulthandler.enable()

from prepare_text_data import Termlist_Generator
from lda_colla_gibbs import LdaSampler


def runtime(last=None):
    now = datetime.now()
    if last is None:
        return now

    else:
        run_time = now-last
        print(run_time)
        if run_time > timedelta(hours=1):
            print('Warning ------ out of time budget')

        return now


def get_p_zw(phi, p_w, p_z):

    p_zw = phi * p_z / p_w  # phi.shape (z, w)  p_w.shape (1, w)  p_z = int

    return p_zw



class NewsProcessor():

    def __init__(self, data_config):
        self.data_config = data_config

    def word_filter(self):
        pass

    def term_document_matrix(self, save_matrix=False):
        print('create td_matrix')
        last = runtime(last=None)

        tg = Termlist_Generator()
        count_vect = CountVectorizer(max_df=self.data_config['max_df'], min_df=self.data_config['min_df'])

        train_doclist = tg.pull_n_docs(self.data_config['train_size'], self.data_config['test_size'])

        td_matrix = count_vect.fit_transform(train_doclist).toarray()  #<class 'scipy.sparse.csr.csr_matrix'>

        print('td matrix done')

        vocab_size = td_matrix.shape[1]
        corpus_size = td_matrix.sum()

        word2id = count_vect.vocabulary_
        for key, value in word2id.items():
            word2id[key] = int(value)

        id2word = {v: k for k, v in word2id.items()}

        df_dict = {}
        for w in range(vocab_size):
            df = np.count_nonzero(td_matrix[:, w])
            df_dict[w] = int(df)

        tf_dict = {}
        tf_array = td_matrix.sum(axis=0)
        p_w = tf_array / corpus_size    # p_w.shape (1, w)
        for i, tf in enumerate(tf_array):
            tf_dict[i] = int(tf)

        corpus_info = {}
        corpus_info['data_config'] = self.data_config
        corpus_info['word2id'] = word2id
        corpus_info['id2word'] = id2word
        corpus_info['vocab_size'] = int(vocab_size)
        corpus_info['corpus_size'] = int(corpus_size)
        corpus_info['tf_dict'] = tf_dict
        corpus_info['df_dict'] = df_dict
        corpus_info['p_w'] = p_w.tolist()



        with open(self.data_config['path'] + 'corpus_info'+ self.data_config['version']+ '.json', 'w', encoding='utf8') as f:
            json.dump(corpus_info, f)
        print('--- corpus_info done')

        if save_matrix is True:
            print('--- save td_matrix')

            np.save(self.data_config['path'] +'td_matrix'+ self.data_config['version'], td_matrix)


            print('--- matrix saved')

        runtime(last=last)

        return td_matrix  # (row = doc, col = word)

    def load_td_matrix(self):
        print('load td_matrix')
        td_matrix = np.load(self.data_config['path'] + 'td_matrix' + self.data_config['version']+'.npy')

        return td_matrix

    def load_corpus_info(self):
        print('load corpus info')
        with open(self.data_config['path'] + 'corpus_info'+ self.data_config['version']+ '.json', 'r', encoding='utf8') as f:
            corpus_info = json.load(f)

        return corpus_info



class TrainFactor():

    def __init__(self, train_plan, data_config):
        self.data_config = data_config
        self.train_plan = train_plan
        self.maxiter = train_plan['maxiter']


    def _initialize(self, corpus_info):
        self.p_w = corpus_info['p_w']
        self.p_z = 1 / self.train_plan['n_topics']

        self.maxlike = -1*10**100
        self.opt_iter = 0

        self.opt_phi = np.arange(1)
        self.opt_nmz = np.arange(1)
        self.opt_nzw = np.arange(1)  # p(z|w)

        self.phi_in_iters = {}
        self.nmz_in_iters = {}
        self.nzw_in_iters = {}
        self.p_zw_in_iters = {}  # p(z|w)

        self.likelihood_in_iters = {}



    def _add_lda_prior(self, alpha, beta, n_topics):
        self.alpha = alpha
        self.beta = beta
        self.n_topics = n_topics


    def _add_sampler(self):
        sampler = LdaSampler(n_topics=self.n_topics, alpha=self.alpha, beta=self.beta)

        return sampler

    def _pack_lda_model(self, save=False):
        model = {}
        # optimal
        model['maximum_likelihood'] = self.maxlike #int
        model['opt_iter'] = self.opt_iter
        model['opt_phi'] = self.opt_phi.tolist()
        model['opt_nmz'] = self.opt_nmz.tolist()
        model['opt_nzw'] = self.opt_nzw.tolist()
        model['opt_p_zw'] = self.opt_p_zw.tolist()

        # every iterations
        model['likelihood_in_iters'] = self.likelihood_in_iters

        # every N iterations
        model['phi_in_iters'] = self.phi_in_iters
        model['nmz_in_iters'] = self.nmz_in_iters
        model['nzw_in_iters'] = self.nzw_in_iters
        model['p_zw_in_iters'] = self.p_zw_in_iters

        if save is True:
            print('--- save model')
            with open(self.data_config['path'] + 'lda_model'+ self.train_plan['version']+ '.json', 'w', encoding='utf8') as f:
                json.dump(model, f)
            print('--- model saved')

        return model


    def build_lda_model(self, td_matrix, corpus_info, alpha, beta, n_topics, save=False, graph=False):
        last = runtime(last=None)
        print('build model')

        self._initialize(corpus_info)

        self._add_lda_prior(alpha=alpha, beta=beta, n_topics=n_topics)

        sampler = self._add_sampler()

        # extract 1. phi   2. nmz   3. nzw from lda_colla_gibbs every iteration
        for i, phi_nmz_nzw in enumerate(sampler.run(matrix=td_matrix, maxiter=self.maxiter)):
            like = sampler.loglikelihood()

            self.likelihood_in_iters[i] = like

            # save distribution & assignment at save_point
            # if i in self.train_plan['save_point']:
            #     self.phi_in_iters[i] = phi_nmz_nzw[0].tolist()
            #     self.nmz_in_iters[i] = phi_nmz_nzw[1].tolist()
            #     self.nzw_in_iters[i] = phi_nmz_nzw[2].tolist()
            #     self.p_zw_in_iters[i] = get_p_zw(phi_nmz_nzw[0], self.p_w, self.p_z).tolist()

            # update maximum likelihood
            if like > self.maxlike:
                self.maxlike = like
                self.opt_iter = i
                self.opt_phi = phi_nmz_nzw[0]
                self.opt_nmz = phi_nmz_nzw[1]
                self.opt_nzw = phi_nmz_nzw[2]

        self.opt_p_zw = get_p_zw(self.opt_phi, self.p_w, self.p_z)

        runtime(last)
        print('--- model done')

        print('---- config ----')
        print('alpha : '+str(alpha))
        print('beta : '+str(beta))
        print('n_topics : '+str(n_topics))
        print('---- optimal ----')
        print('maximum likelihood : '+str(self.maxlike))
        print('opt_iter '+str(self.opt_iter))


        # model = self._pack_lda_model(save=save)

        if graph is True:
            iter = list(self.likelihood_in_iters.keys())
            likelihood = list(self.likelihood_in_iters.values())

            plt.plot(iter, likelihood)
            plt.xlabel('iteration')
            plt.ylabel('loglikelihood')
            plt.show()

        return self.opt_p_zw, self.maxlike


    def _tt_similarity_matrix(self, opt_p_zw, save=False):
        # p_zw = np.array(model['opt_p_zw']).T   # opt_pzw (z, w)  p_zw (w, z)

        p_zw = opt_p_zw.T

        tt_similarity_matrix = cosine_similarity(p_zw, p_zw)

        if save is True:
            np.save(self.data_config['path'] + 'tt_similarity_matrix' + self.train_plan['version'], tt_similarity_matrix)

        return tt_similarity_matrix


    def find_word_with_sim(self, tt_sim_matrix, corpus_info, word, n_target=10):
        word2id = corpus_info['word2id']
        id2word = corpus_info['id2word']
        wordlist = []
        if word in word2id.keys():
            wordid = word2id[word]

            tt_sim = tt_sim_matrix[wordid, :]
            top_wordid = np.argsort(tt_sim)[-1*n_target:][::-1]
            for i in list(top_wordid):
                wordlist.append(id2word[str(i)])

            return wordlist
        else:
            print('word not in vocab')
            return wordlist


    def evaluate_model(self, opt_p_zw, ground_truth, corpus_info, window=100, save=False):
        print('evaluate model')
        last = runtime(last=None)

        tt_sim_matrix = self._tt_similarity_matrix(opt_p_zw)

        score_dict = {}
        score = []
        for key, value in ground_truth.items():
            correct = 0
            exam = 0
            for word in value:
                others = list(set(value)-set(word))
                predict = self.find_word_with_sim(tt_sim_matrix, corpus_info, word, n_target=window)
                for other_word in others:
                    exam += 1
                    if other_word in predict:
                        correct += 1

            accuracy = correct / exam
            score.append(accuracy)
            score_dict[key] = accuracy

        avg_score = np.mean(score)
        score_dict['avg_score'] = avg_score

        runtime(last=last)

        print(score_dict)
        print(avg_score)

        if save is True:
            with open(self.data_config['path'] + 'evaluation'+ self.train_plan['version']+ '.json', 'w', encoding='utf8') as f:
                json.dump(score_dict, f)

        return avg_score



class Tune_Alpha_Beta():
    def __init__(self, data_config, train_plan, init_alpha, init_beta):
        self.data_config = data_config
        self.train_plan = train_plan

        # prepare data
        news = NewsProcessor(data_config)
        self.td_matrix = news.load_td_matrix()
        self.corpus_info = news.load_corpus_info()

        with open(data_config['path'] + 'ground_truth'+ data_config['version']+ '.json', 'r', encoding='utf8') as f:
            self.ground_truth = json.load(f)


        # initialize
        self.n_topics = 20

        self.step = train_plan['step']

        self.last_alpha_used_in_beta = None
        self.last_beta_used_in_alpha = None
        self.last_alpha_move = None
        self.last_beta_move = None

        self.best_score = 0
        self.best_maxlike = -1*10**100

        self.alpha_beta_recorder = []

        # start from init
        self.set_alpha(init_alpha, init_beta)


    def score_watcher(self, new_score):
        if new_score > self.best_score:
            score_report = 'better'
            self.best_score = new_score
        else:
            score_report = 'worse'

        return score_report

    def maxlike_watcher(self, new_maxlike):
        if new_maxlike > self.best_maxlike:
            maxlike_report = 'better'
            self.best_maxlike = new_maxlike

        else:
            maxlike_report = 'worse'

        return maxlike_report



    def beta_param_watcher(self, alpha):
        if self.last_alpha_used_in_beta is None:
            param_report = 'none'


        elif alpha != self.last_alpha_used_in_beta:
            param_report = 'pro'

        else:
            param_report = 'no_pro'

        return param_report



    def alpha_param_watcher(self, beta):
        if self.last_beta_used_in_alpha is None:
            param_report = 'none'

        elif beta != self.last_beta_used_in_alpha:
            param_report = 'pro'

        else:
            param_report = 'no_pro'

        return param_report



    def set_alpha(self, alpha, beta):
        print('----------')
        print('set alpha')
        print('alpha : ' + str(alpha) + '  beta: ' + str(beta))
        self.alpha_beta_recorder.append([alpha,beta])

        # deicide how to run the next model , by last_move, param_report
        param_report = self.alpha_param_watcher(beta=beta)
        if param_report == 'none':
            side = 'both'

        elif param_report == 'pro':
            side = 'both'

        elif param_report == 'no_pro' and self.last_alpha_move == 'up':
            side = 'up'

        elif param_report == 'no_pro' and self.last_alpha_move == 'down':
            side = 'down'

        # receive the instruction
        train = TrainFactor(self.train_plan, self.data_config)

        if side == 'both':
            alpha_up = alpha + self.step
            model_alpha_up, maxlike_alpha_up = train.build_lda_model(td_matrix=self.td_matrix, corpus_info=self.corpus_info, alpha=alpha_up, beta=beta, n_topics=self.n_topics)
            score_alpha_up = train.evaluate_model(model_alpha_up, self.ground_truth, self.corpus_info)
            gc.collect()

            alpha_down = alpha - self.step
            model_alpha_down, maxlike_alpha_down = train.build_lda_model(td_matrix=self.td_matrix, corpus_info=self.corpus_info, alpha=alpha_down, beta=beta, n_topics=self.n_topics)
            score_alpha_down = train.evaluate_model(model_alpha_down, self.ground_truth, self.corpus_info)
            gc.collect()

            if score_alpha_up > score_alpha_down:
                new_alpha = alpha_up
                new_score = score_alpha_up
                new_maxlike = maxlike_alpha_up
                side = 'up'

            else:
                new_alpha = alpha_down
                new_score = score_alpha_down
                new_maxlike = maxlike_alpha_down
                side = 'down'

        elif side == 'up':
            new_alpha = alpha + self.step
            new_model, new_maxlike = train.build_lda_model(td_matrix=self.td_matrix, corpus_info=self.corpus_info, alpha=new_alpha, beta=beta, n_topics=self.n_topics)
            new_score = train.evaluate_model(new_model, self.ground_truth, self.corpus_info)


        elif side == 'down':
            new_alpha = alpha - self.step
            new_model, new_maxlike = train.build_lda_model(td_matrix=self.td_matrix, corpus_info=self.corpus_info, alpha=new_alpha, beta=beta, n_topics=self.n_topics)
            new_score = train.evaluate_model(new_model, self.ground_truth, self.corpus_info)



        # decide what to run next , by param, score  / record param_move this time
        score_report = self.score_watcher(new_score)
        print(score_report)
        param_report = self.alpha_param_watcher(beta=beta)
        print(param_report)
        self.last_beta_used_in_alpha = beta

        if score_report == 'better':
            if param_report == 'none' or param_report == 'no_pro':
                self.last_alpha_move = side
                print('new score : ' + str(new_score))
                print('new_maxlike : ' + str(new_maxlike))

                return self.set_alpha(alpha=new_alpha, beta=beta)

            elif param_report == 'pro':
                self.last_alpha_move = side
                print('new score : ' + str(new_score))
                print('new_maxlike : ' + str(new_maxlike))

                self.set_beta(alpha=new_alpha, beta=beta)

        elif score_report == 'worse':
            print('lstay')

            self.set_beta(alpha=alpha, beta=beta)

        else:
            print('WTF?')




    def set_beta(self, alpha, beta):
        print('----------')
        print('set beta')
        print('alpha : '+str(alpha)+'  beta: '+str(beta))
        self.alpha_beta_recorder.append([alpha, beta])

        # deicide how to run the next model , by last_move, param_report
        param_report = self.beta_param_watcher(alpha=alpha)
        if param_report == 'none':
            side = 'both'

        elif param_report == 'pro':
            side = 'both'

        elif param_report == 'no_pro' and self.last_beta_move == 'up':
            side = 'up'

        elif param_report == 'no_pro' and self.last_beta_move == 'down':
            side = 'down'


        # receive the instruction
        train = TrainFactor(self.train_plan, self.data_config)

        if side == 'both':
            beta_up = beta + self.step
            model_beta_up, maxlike_beta_up = train.build_lda_model(td_matrix=self.td_matrix, corpus_info=self.corpus_info, alpha=alpha, beta=beta_up, n_topics=self.n_topics)
            score_beta_up = train.evaluate_model(model_beta_up, self.ground_truth, self.corpus_info)
            gc.collect()

            beta_down = beta - self.step
            model_beta_down, maxlike_beta_down = train.build_lda_model(td_matrix=self.td_matrix, corpus_info=self.corpus_info, alpha=alpha, beta=beta_down, n_topics=self.n_topics)
            score_beta_down = train.evaluate_model(model_beta_down, self.ground_truth, self.corpus_info)
            gc.collect()

            if maxlike_beta_up > maxlike_beta_down:
                new_beta = beta_up
                new_maxlike = maxlike_beta_up
                new_score = score_beta_up
                side = 'up'

            else:
                new_beta = beta_down
                new_maxlike = maxlike_beta_down
                new_score = score_beta_down
                side = 'down'

        elif side == 'up':
            new_beta = beta + self.step
            new_model, new_maxlike = train.build_lda_model(td_matrix=self.td_matrix, corpus_info=self.corpus_info, alpha=alpha, beta=new_beta, n_topics=self.n_topics)
            new_score = train.evaluate_model(new_model, self.ground_truth, self.corpus_info)
            gc.collect()

        elif side == 'down':
            new_beta = beta - self.step
            new_model, new_maxlike = train.build_lda_model(td_matrix=self.td_matrix, corpus_info=self.corpus_info, alpha=alpha, beta=new_beta, n_topics=self.n_topics)
            new_score = train.evaluate_model(new_model, self.ground_truth, self.corpus_info)
            gc.collect()



        # decide what to run next , by param, maxlike  / record param_move this time
        maxlike_report = self.maxlike_watcher(new_maxlike)
        param_report = self.beta_param_watcher(alpha=alpha)
        self.last_alpha_used_in_beta = alpha

        if maxlike_report == 'better':
            self.last_beta_move = side
            print('new score : ' + str(new_score))
            print('new_maxlike : ' + str(new_maxlike))

            return self.set_beta(alpha=alpha, beta=new_beta)


        elif maxlike_report == 'worse':
            if param_report == 'pro':
                print('stay')
                self.set_alpha(alpha=alpha, beta=beta)

            if param_report == 'no_pro' or param_report == 'none':
                print('end of tuning')
                print('alpha : ' + str(alpha))
                print('beta : ' + str(beta))
                print('new score : ' + str(new_score))
                print('new_maxlike : ' + str(new_maxlike))

                with open(self.data_config['path'] + 'alpha_beta_record' + self.train_plan['version'] + '.json', 'w',
                          encoding='utf8') as f:
                    json.dump(self.alpha_beta_recorder, f)
        else:
            print('WTF?')


