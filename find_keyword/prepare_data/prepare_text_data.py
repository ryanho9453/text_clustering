from pymongo import MongoClient
import jieba
import re
import os
import random


class StopWords:
    def __init__(self):
        marks = ['（', '〔', '［', '｛', '《', '【', '〖', '〈', '(', '[' '{', '<',
                 '）', '〕', '］', '｝', '》', '】', '〗', '〉', ')', ']', '}', '>',
                 '“', '‘', '『', '』', '。', '？', '?', '！', '!', '，', ',', '', '；',
                 ';', '、', '：', ':', '……', '…', '——', '—', '－－', '－', '-', ' ',
                 '「', '」', '／', '/', ',', '.', '=', '+', '#', '\xa0', '\r\n', '@', '...', '\t',
                 '|', '%', '#', 'of', 'in', 'rss']

        script_path = os.path.dirname(__file__)

        with open(os.path.join(script_path, 'stopwords/stopword_cht.txt'), 'r') as a:
            stopwords_cht = [word.strip('\n') for word in a]

        with open(os.path.join(script_path, 'stopwords/terrier-stop.txt'), 'r') as a:
            stopwords_eng = [word.strip('\n') for word in a]

        self.stopwords = marks + stopwords_cht + stopwords_eng


class MongoDownloader:
    def __init__(self,  random_pull=False, headline=False):
        self.client = MongoClient('localhost', 27017)
        self.random_pull = random_pull
        self.headline = headline

    def pull_n_docs(self, train_size, test_size):
        if self.headline:
            news_dict = self.client['focal']['news_data'].find({}, {'head': True, 'context': True})

        else:
            news_dict = self.client['focal']['news_data'].find({}, {'context': True})

        # news is dict before list()
        news_list = list(news_dict)

        if self.random_pull:
            random.shuffle(news_list)

        train_news = news_list[0: train_size]
        test_news = news_list[train_size: train_size + test_size]

        return train_news, test_news


class DoclistGenerator:
    def __init__(self):
        stop = StopWords()
        self.stopwords = stop.stopwords

    def gen_n_docs(self, train_size, test_size, random_pull=False):

        mongo = MongoDownloader(random_pull=random_pull, headline=False)
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
        for news_dict in docs:
            seglist = jieba.cut(news_dict['context'])
            string = ''
            for term in seglist:
                term = term.lower()
                if len(term) > 1 and not re.search('[0-9]',term):
                    if term not in self.stopwords:
                        if string != '':
                            string += ' '
                            string += str(term)
                        else:
                            string += str(term)
            doclist.append(string)

        return doclist
