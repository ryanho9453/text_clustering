import jieba
import re
import os
import sys
sys.path.append('/Users/ryanho/Documents/python/focal/text_clustering/find_keyword/prepare_data/')

from mongo import Mongo

script_path = os.path.dirname(__file__)
whitelist_path = os.path.join(script_path, 'whitelist.txt')
jieba.load_userdict(whitelist_path)


class StopWords:
    def __init__(self):
        marks = ['（', '〔', '［', '｛', '《', '【', '〖', '〈', '(', '[' '{', '<',
                 '）', '〕', '］', '｝', '》', '】', '〗', '〉', ')', ']', '}', '>',
                 '“', '‘', '『', '』', '。', '？', '?', '！', '!', '，', ',', '', '；',
                 ';', '、', '：', ':', '……', '…', '——', '—', '－－', '－', '-', ' ',
                 '「', '」', '／', '/', ',', '.', '=', '+', '#', '\xa0', '\r\n', '@', '...', '\t', '\n',
                 '|', '%', '#', 'of', 'in', 'rss']

        with open(os.path.join(script_path, 'stopwords/stopword_cht.txt'), 'r') as a:
            stopwords_cht = [word.strip('\n') for word in a]

        with open(os.path.join(script_path, 'stopwords/terrier-stop.txt'), 'r') as a:
            stopwords_eng = [word.strip('\n') for word in a]

        self.stopwords = marks + stopwords_cht + stopwords_eng


class DoclistGenerator:
    def __init__(self):
        stop = StopWords()
        self.stopwords = stop.stopwords

    def gen_n_docs(self, train_size, test_size, random_pull=False):

        mongo = Mongo(random_pull=random_pull, headline=False)
        train_news, test_news = mongo.pull_n_docs(train_size, test_size)

        train_doclist = self._doc2terms(train_news)
        test_doclist = self._doc2terms(test_news)

        return train_doclist, test_doclist

    def _doc2terms(self, docs):
        """
        :param docs: list of doc (doc is type dict)
        :return: list of doc (do  is type string)

        為了用 Countvectorizer 將 docs 轉成 term_document matrix
        將 '我要成為電腦工程師' 轉成 string '我 要 成為 電腦 工程師' , 在收集成 list

        """
        doclist = []
        whitelist = self._read_whitelist(whitelist_path)
        for news_dict in docs:
            seglist = jieba.cut(news_dict['context'])
            string = ''
            for term in seglist:
                term = term.lower()
                if re.search('[0-9]', term):
                    if term in whitelist and term not in self.stopwords:
                        if string != '':
                            string += ' '
                            string += str(term)
                        else:
                            string += str(term)

                else:
                    if term not in self.stopwords:
                        if string != '':
                            string += ' '
                            string += str(term)
                        else:
                            string += str(term)

            doclist.append(string)

        return doclist

    def _read_whitelist(self, whitelist_path):
        with open(whitelist_path, 'r') as f:
            lines = f.readlines()

        termlist = list()
        for line in lines:
            newline = line.replace('\n', '')
            termlist.append(newline)

        return termlist
