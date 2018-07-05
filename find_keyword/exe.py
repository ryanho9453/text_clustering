import json
import numpy as np
import os
import sys
from pprint import pprint

sys.path.append('/Users/ryanho/Documents/python/focal/text_clustering/find_keyword/prepare_data/')
sys.path.append('/Users/ryanho/Documents/python/focal/text_clustering/find_keyword/lda_collapsed_gibbs/')

from word_predictor import WordPredictor
from prior_cluster_editor import ClusterEditor
from model import LdaModel
from corpus import Corpus

with open('/Users/ryanho/Documents/python/focal/text_clustering/find_keyword/config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

cluster_editor = ClusterEditor(config)

keyword = ['富士康', 'toyota']

""" find similar words of keywords """

# word_predictor = WordPredictor(config)
# results = word_predictor.find_similar_word(keyword, n_target=15)
# pprint(results)

""" show word topic distribution P( z | w ) """

# model = LdaModel(config)
# model.show_word_topic_distribution(keyword)


""" show word clusters """
# clusters = cluster_editor.get_cluster()
# pprint(clusters)


""" find similar word for words in cluster """
# results = cluster_editor.find_similar_word_for_cluster(cluster_no=0, n_newword=5)
# pprint(results)


""" reccomend new word to add in cluster (excluding words already in cluster) """
# results = cluster_editor.find_new_word_for_cluster(cluster_no=0, n_newword=5)
# pprint(results)
