"""
create term document matrix
at the same time, collect information about the corpus and the word_id_converter
and then save to local as 'corpus_info.json' , 'word_id_converter.json'

also for lda calculation, calculate p(w) and save to local as numpy 'p_w.np'

word_id_converter{
                  'word2id': {word: id},
                  'id2word': {id: word}
                  }

td_matrix.shape( n_docs, n_words)

"""

import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer

from .prepare_text_data import DoclistGenerator


class TermDocumentMatrix:

    def __init__(self, config):
        self.config = config

    def create(self, save_matrix=False):
        dg = DoclistGenerator()
        count_vect = CountVectorizer(max_df=self.config['max_df'], min_df=self.config['min_df'])

        print('create term_document matrix')

        train_doclist, test_doclist = dg.gen_n_docs(self.config['train_size'], self.config['test_size'],
                                                    random_pull=False)

        # train news only
        # CountVectorizer deal with list of string , english format doc
        td_matrix = count_vect.fit_transform(train_doclist).toarray()   # <class 'scipy.sparse.csr.csr_matrix'>

        if save_matrix is True:
            print('save td_matrix')
            np.save(self.config['path'] + 'td_matrix.npy', td_matrix)

        vocab_size, corpus_size = self._collect_corpus_info(td_matrix)
        self._create_word_id_converter(count_vect)
        self._calculate_p_w(td_matrix, corpus_size)

        return td_matrix     # (row = doc, col = word)

    def get_corpus_info(self):
        return self.corpus_info

    def load(self):
        print('load matrix')
        td_matrix = np.load(self.config['path'] + 'td_matrix.npy')

        return td_matrix

    def _collect_corpus_info(self, td_matrix):

        vocab_size = td_matrix.shape[1]
        corpus_size = td_matrix.sum()

        # dict key is wordid
        df_dict = self.__gen_df_dict(td_matrix, vocab_size)
        tf_dict = self.__gen_tf_dict(td_matrix)

        self.corpus_info = dict()
        self.corpus_info['config'] = self.config
        self.corpus_info['vocab_size'] = int(vocab_size)
        self.corpus_info['corpus_size'] = int(corpus_size)
        self.corpus_info['df_dict'] = df_dict
        self.corpus_info['tf_dict'] = tf_dict

        with open(self.config['path'] + 'corpus_info.json', 'w', encoding='utf8') as f:
            json.dump(self.corpus_info, f)

        return vocab_size, corpus_size

    def _create_word_id_converter(self, count_vect):
        word_id_converter = {}
        word2id = count_vect.vocabulary_
        for key, value in word2id.items():
            word2id[key] = int(value)
        id2word = {v: k for k, v in word2id.items()}
        word_id_converter['word2id'] = word2id
        word_id_converter['id2word'] = id2word

        with open(self.config['path'] + 'word_id_converter.json', 'w', encoding='utf8') as f:
            json.dump(word_id_converter, f)

    def _calculate_p_w(self, td_matrix, corpus_size):
        tf_array = td_matrix.sum(axis=0)
        p_w = tf_array / corpus_size  # p_w.shape (1, w)

        np.save(self.config['path'] + 'p_w.npy', p_w)

    def __gen_df_dict(self, td_matrix, vocab_size):
        df_dict = {}
        for w in range(vocab_size):
            df = np.count_nonzero(td_matrix[:, w])
            df_dict[w] = int(df)

        return df_dict

    def __gen_tf_dict(self, td_matrix):
        tf_dict = {}
        tf_array = td_matrix.sum(axis=0)
        for i, tf in enumerate(tf_array):
            tf_dict[i] = int(tf)

        return tf_dict
