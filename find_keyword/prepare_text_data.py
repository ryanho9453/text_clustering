from pymongo import MongoClient
import jieba
import re
import numpy as np
from gensim.models.doc2vec import TaggedDocument
import json



class StopWords():
    def __init__(self):
        marks = ['（', '〔', '［', '｛', '《', '【', '〖', '〈', '(', '[' '{', '<',
                     '）', '〕', '］', '｝', '》', '】', '〗', '〉', ')', ']', '}', '>',
                     '“', '‘', '『', '』', '。', '？', '?', '！', '!', '，', ',', '', '；',
                     ';', '、', '：', ':', '……', '…', '——', '—', '－－', '－', '-', ' ',
                     '「', '」', '／', '/', ',', '.', '=', '+', '#', '\xa0', '\r\n', '@', '...', '\t',
                     '|', '%', '#', 'of', 'in', 'rss']

        with open('stopwords/stopword_cht.txt', 'r') as a:
            stopwords_cht = [word.strip('\n') for word in a]

        with open('stopwords/terrier-stop.txt', 'r') as a:
            stopwords_eng = [word.strip('\n') for word in a]

        self.stopwords = marks + stopwords_cht + stopwords_eng


class Termlist_Generator():
    def __init__(self):
        stop = StopWords()
        self.stopwords = stop.stopwords

    def pull_cate(self,category,directory,jsonname, train_size, test_size):
        client = MongoClient('localhost', 27017)
        cate_news_dic = client['focal']['news_data'].find({'category':category},{'keyword':True,'context':True})
        cate_news_dic_C = list(cate_news_dic)

        cate_news_collect = []
        for news_dic in cate_news_dic_C:
            cate_news_collect.append(news_dic['context'])

        train_data = cate_news_collect[0:train_size]
        test_data = cate_news_collect[train_size:train_size+test_size]
        print(len(train_data))

        train_termlists = []
        for news in train_data:
            seglist = jieba.cut(news)
            termlist = []
            for term in seglist:
                term = term.lower()
                if len(term) > 1 and not re.search('[0-9]',term):
                    if term not in self.stopwords:
                        termlist.append(term)
            train_termlists.append(termlist)
        with open(directory + jsonname + '.json', 'w', encoding='utf8') as f:
            json.dump(train_termlists, f)

        test_termlists = []
        for news in test_data:
            seglist = jieba.cut(news)
            termlist = []
            for term in seglist:
                term = term.lower()
                if len(term) > 1 and not re.search('[0-9]',term):
                    if term not in self.stopwords:
                        termlist.append(term)
            test_termlists.append(termlist)
        with open(directory + jsonname + 't.json','w') as f:
            json.dump(train_termlists, f)

        return train_termlists

    def pull_n_docs(self, train_size, test_size):
        client = MongoClient('localhost', 27017)
        mongo_news = client['focal']['news_data'].find({}, {'context': True})
        news_dict_C = list(mongo_news)

        train_data = news_dict_C[0:train_size]
        test_data = news_dict_C[train_size:train_size+test_size]

        train_doclist = []
        for news_dict in train_data:
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
            train_doclist.append(string)

        test_doclist = []
        for news_dict in test_data:
            seglist = jieba.cut(news_dict['context'])
            string = ''
            for term in seglist:
                term = term.lower()
                if len(term) > 1 and not re.search('[0-9]', term):
                    if term not in self.stopwords:
                        if string != '':
                            string += ' '
                            string += str(term)
                        else:
                            string += str(term)
            test_doclist.append(string)


        return train_doclist, test_doclist

    def pull_random_doc(self,train_size, size=1):
        client = MongoClient('localhost', 27017)
        mongo_news = client['focal']['news_data'].find({}, {'context': True})
        news_dict_C = list(mongo_news)

        start = int(np.random.randint(train_size))

        doc = news_dict_C[start:start+size]

        return doc

    def pull_all(self,directory, jsonname):
        client = MongoClient('localhost', 27017)
        cate_news_dic = client['focal']['news_data'].find({}, {'context': True})
        cate_news_dic_C = list(cate_news_dic)

        context_collect = []
        for news_dic in cate_news_dic_C:
            context = news_dic['context']
            seg_context = jieba.cut(context)
            t_list = []
            for j in seg_context:
                j = j.lower()
                if j not in self.stopwords:
                    t_list.append(j)
            context_collect.append(t_list)

        with open(directory + 'alldocs' + jsonname + '.json', 'w') as f:
            json.dump(context_collect, f)

        return context_collect

    def pull_all_headntext(self, directory, jsonname):
        client = MongoClient('localhost', 27017)
        cate_news_dic = client['focal']['news_data'].find({}, {'head': True, 'context': True})
        cate_news_dic_C = list(cate_news_dic)

        headntext_collect = []
        tagdoc_collect = []
        for news_dic in cate_news_dic_C:
            context = news_dic['context']
            head = news_dic['head']
            seg_context = jieba.cut(context)
            t_list = []
            for j in seg_context:
                j = j.lower()
                if j not in self.stopwords:
                    t_list.append(j)

            tagdoc = TaggedDocument(t_list, [head])
            tagdoc_collect.append(tagdoc)

            headntext = [head, t_list]
            headntext_collect.append(headntext)

        print(tagdoc_collect)
        with open(directory + 'headntext' + jsonname + '.json', 'w') as f:
            json.dump(headntext_collect, f)

        return tagdoc_collect



    def pull_docs_with_headline(self,category,directory,jsonname):
        client = MongoClient('localhost', 27017)
        cate_news_dic = client['focal']['news_data'].find({'category': category}, {'head': True, 'context': True})
        cate_news_dic_C = list(cate_news_dic)

        headntext_collect = []
        tagdoc_collect = []
        for news_dic in cate_news_dic_C:
            context = news_dic['context']
            head = news_dic['head']
            seg_context = jieba.cut(context)
            t_list = []
            for j in seg_context:
                j = j.lower()
                if j not in self.stopwords:
                    t_list.append(j)
            # print(h_list,t_list)

            tagdoc = TaggedDocument(t_list,[head])
            tagdoc_collect.append(tagdoc)


            headntext = [head,t_list]
            headntext_collect.append(headntext)

        print(tagdoc_collect)
        with open(directory + 'docs'+jsonname + '.json', 'w') as f:
            json.dump(headntext_collect, f)

        return tagdoc_collect


    def pull_docs_with_keyword(docs, keyword,path,name,a,b,lowerbound):
        termlist_C = []
        for termlist in docs:
            if len(set(termlist).intersection(set(keyword))) > lowerbound:
                    termlist_C.append(termlist)
        print(len(termlist_C))

        termlist_C = termlist_C[a:b]

        with open(path+name +'.json','w') as f:
            json.dump(termlist_C, f)
        return termlist_C




