from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json


class Word_Predictor():
    def __init__(self, config, term_feature_maxtrix):
        self.config = config

        self.tt_sim_matrix = self._tt_similarity_matrix(term_feature_maxtrix)
        with open(self.config['path'] + 'word_id_converter.json', 'r', encoding='utf8') as f:
            self.word_id_converter = json.load(f)


    def _tt_similarity_matrix(self, term_feature_maxtrix, save=False):
        """
        term to term similarity matrix

        use terms' feature to calcualte cosine similarity

        """
        tt_similarity_matrix = cosine_similarity(term_feature_maxtrix, term_feature_maxtrix)

        if save is True:
            np.save(self.config['path'] + 'tt_similarity_matrix' + self.config['model_ver'] + '.npy', tt_similarity_matrix)

        return tt_similarity_matrix


    def find_similar_word(self, word, n_target=10):
        word2id = self.word_id_converter['word2id']
        id2word = self.word_id_converter['id2word']
        wordlist = []
        if word in word2id.keys():
            wordid = word2id[word]
            tt_sim = self.tt_sim_matrix[wordid, :]
            top_wordid = np.argsort(tt_sim)[- 1 * n_target:][::-1]
            for i in list(top_wordid):
                wordlist.append(id2word[str(i)])

            return wordlist
        else:

            return wordlist