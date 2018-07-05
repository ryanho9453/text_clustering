import numpy as np
import json
from pprint import pprint

"""
with given word, find similar words
use cosine similarity

"""


class WordPredictor:
    def __init__(self, config, term_similarity_matrix=True):
        self.config = config

        if term_similarity_matrix:
            self.t_sim_matrix = np.load(self.config['path'] + 'term_similarity_matrix' + self.config['model_ver'] + '.npy')

        with open(self.config['path'] + 'word_id_converter.json', 'r', encoding='utf8') as f:
            self.word_id_converter = json.load(f)

    def find_similar_word(self, word, n_target=10):
        """
        :param n_target: number of similar words return

        """
        word2id = self.word_id_converter['word2id']
        id2word = self.word_id_converter['id2word']

        result_dict = dict()

        if type(word) is str:
            if word in word2id.keys():
                wordid = word2id[word]
                tt_sim = self.t_sim_matrix[wordid, :]
                n_target_plus_itself = n_target + 1
                top_wordid = list(np.argsort(tt_sim)[- 1 * n_target_plus_itself:][::-1])
                top_wordid.remove(wordid)

                wordlist = list()
                for i in top_wordid:
                    wordlist.append(id2word[str(i)])

                result_dict[word] = wordlist

                return result_dict

            else:
                print(str(word)+' not in dictionary')

        elif type(word) is list:
            for w in word:
                if w in word2id.keys():
                    wordid = word2id[w]
                    tt_sim = self.t_sim_matrix[wordid, :]
                    n_target_plus_itself = n_target + 1
                    top_wordid = list(np.argsort(tt_sim)[- 1 * n_target_plus_itself:][::-1])
                    if wordid in top_wordid:
                        top_wordid.remove(wordid)

                    wordlist = list()
                    for i in top_wordid:
                        wordlist.append(id2word[str(i)])

                    result_dict[w] = wordlist

                else:
                    print(str(w) + ' not in dictionary')

            return result_dict

        else:
            print('input must be string or list')
