import numpy as np
import json

from prepare_text_data import Termlist_Generator
from sklearn.feature_extraction.text import CountVectorizer




class Term_document_Matrix():

    def __init__(self, config):
        self.config = config


    def create(self, save_matrix=False):
        tg = Termlist_Generator()
        count_vect = CountVectorizer(max_df=self.config['max_df'], min_df=self.config['min_df'])

        print('create td_matrix')

        train_doclist, test_doclist = tg.pull_n_docs(self.config['train_size'], self.config['test_size'])
        td_matrix = count_vect.fit_transform(train_doclist).toarray()  #<class 'scipy.sparse.csr.csr_matrix'>

        if save_matrix is True:
            print('save td_matrix')
            np.save(self.config['path'] + 'td_matrix.npy', td_matrix)

        word_id_converter = {}
        word2id = count_vect.vocabulary_
        for key, value in word2id.items():
            word2id[key] = int(value)
        id2word = {v: k for k, v in word2id.items()}
        word_id_converter['word2id'] = word2id
        word_id_converter['id2word'] = id2word

        self._pack_results(td_matrix, word_id_converter)

        return td_matrix     # (row = doc, col = word)

    def get_corpus_info(self):
        return self.corpus_info


    def load(self):
        print('load matrix')
        td_matrix = np.load(self.config['path'] + 'td_matrix.npy')

        return td_matrix


    def _pack_results(self, td_matrix, word_id_converter):
        # save corpus info
        vocab_size = td_matrix.shape[1]
        corpus_size = td_matrix.sum()

        self.corpus_info = {}
        self.corpus_info['config'] = self.config
        self.corpus_info['vocab_size'] = int(vocab_size)
        self.corpus_info['corpus_size'] = int(corpus_size)

        with open(self.config['path'] + 'corpus_info.json', 'w', encoding='utf8') as f:
            json.dump(self.corpus_info, f)

        # save p_w
        tf_array = td_matrix.sum(axis=0)
        p_w = tf_array / corpus_size     # p_w.shape (1, w)
        np.save(self.config['path'] + 'p_w.npy', p_w)

        # save word_id_converter
        with open(self.config['path'] + 'word_id_converter.json', 'w', encoding='utf8') as f:
            json.dump(word_id_converter, f)


    def __df_dict(self, td_matrix, vocab_size):
        df_dict = {}
        for w in range(vocab_size):
            df = np.count_nonzero(td_matrix[:, w])
            df_dict[w] = int(df)

        return df_dict

    def __tf_dict(self, td_matrix, corpus_size):
        tf_dict = {}
        tf_array = td_matrix.sum(axis=0)
        for i, tf in enumerate(tf_array):
            tf_dict[i] = int(tf)

        return tf_dict


