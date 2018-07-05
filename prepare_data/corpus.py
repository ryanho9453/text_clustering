import json


class Corpus:
    def __init__(self, config):
        self.config = config

    def check_word_in_vocabulary(self, wordlist):
        with open(self.config['path'] + 'word_id_converter.json', 'r', encoding='utf8') as f:
            word_id_converter = json.load(f)

        word2id = word_id_converter['word2id']

        result = {'in': [],
                  'not_in': []}
        for word in wordlist:
            if word in word2id.keys():
                result['in'].append(word)

            else:
                result['not_in'].append(word)

        print(str(len(wordlist)) + ' words')
        print(str(len(result['in'])) + ' in')
        print(str(len(result['not_in'])) + ' not in')

    def show_word_df_tf(self, wordlist):
        with open(self.config['path'] + 'corpus_info.json', 'r', encoding='utf8') as f:
            corpus_info = json.load(f)

        df_dict = corpus_info['df_dict']
        tf_dict = corpus_info['tf_dict']

        word_info = dict()
        unknown_word = []
        for word in wordlist:
            if word in df_dict.keys():
                word_info[word] = {'df': df_dict[word], 'tf': tf_dict[word]}

            else:
                unknown_word.append(word)

        print('unknown word :'+str(unknown_word))
        print(word_info)

    def get_word2id(self):
        with open(self.config['path'] + 'word_id_converter.json', 'r', encoding='utf8') as f:
            word_id_converter = json.load(f)

        word2id = word_id_converter['word2id']

        return word2id

    def get_id2word(self):
        with open(self.config['path'] + 'word_id_converter.json', 'r', encoding='utf8') as f:
            word_id_converter = json.load(f)

        id2word = word_id_converter['id2word']

        return id2word
