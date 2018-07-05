from pymongo import MongoClient
import random


class Mongo:
    def __init__(self,  random_pull=False, headline=False):
        connection = MongoClient('localhost', 27017)
        admin = connection['admin']
        admin.authenticate('rrpdream', 'rrpdream9453')
        self.db = connection['KeywordFinder']

        self.random_pull = random_pull
        self.headline = headline

    def pull_n_docs(self, train_size, test_size):
        if self.headline:
            news_dict = self.db['news_data'].find({}, {'head': True, 'context': True})

        else:
            news_dict = self.db['news_data'].find({}, {'context': True})

        # news is dict before list()
        news_list = list(news_dict)

        if self.random_pull:
            random.shuffle(news_list)

        train_news = news_list[0: train_size]
        test_news = news_list[train_size: train_size + test_size]

        return train_news, test_news

    def pull_project_keywords(self, project_name):
        keyword_document = self.db['project_keywords'].find({'project_name': project_name}, {'keyword': True})

        keyword_dict = list(keyword_document)[0]
        keyword_list = keyword_dict['keyword']

        return keyword_list
