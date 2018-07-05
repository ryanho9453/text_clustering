from datetime import datetime
from pymongo import MongoClient
import json
import sys

package_path = '/Users/ryanho/Documents/python/focal/text_clustering/find_keyword/'
modules = ['lda_collapsed_gibbs/', 'parameter_learn/', 'prepare_data/', 'tools/']
for module in modules:
    sys.path.append(package_path+module)

from term_document_matrix import TermDocumentMatrix
from model import LdaModel
from word_predictor import WordPredictor
from evaluate import Evaluate
from term_similarity_matrix import TermSimilarityMatrix
from climber import Climber


class Flows:
    def __init__(self, config):
        self.config = config

    def prepare_data(self):
        matrix = TermDocumentMatrix(self.config)
        matrix.create(save_matrix=True)

    def prepare_build_and_evaluate(self):
        matrix = TermDocumentMatrix(self.config)
        td_matrix = matrix.create(save_matrix=True)

        model = LdaModel(self.config)
        model.build(td_matrix, self.config['alpha'], self.config['beta'], self.config['n_topics'], save_model=True)
        p_zw = model.get_p_zw()

        term_similarity_matrix = TermSimilarityMatrix(self.config)
        term_similarity_matrix.create(p_zw, save=True)

        predictor = WordPredictor(self.config)

        evaluate = Evaluate(self.config)
        evaluate.ground_truth(predictor)

    def build_and_evaluate(self):
        matrix = TermDocumentMatrix(self.config)
        td_matrix = matrix.load()

        model = LdaModel(self.config)
        model.build(td_matrix, self.config['alpha'], self.config['beta'], self.config['n_topics'], save_model=True)
        p_zw = model.get_p_zw()

        term_similarity_matrix = TermSimilarityMatrix(self.config)
        term_similarity_matrix.create(p_zw, save=True)

        predictor = WordPredictor(self.config)

        evaluate = Evaluate(self.config)
        evaluate.ground_truth(predictor)

    def tune_alpha_beta(self):
        climber = Climber(self.config)
        climber.toward_summit()

    def predict_project_terms_in_mongo(self, project_name, output):
        connection = MongoClient('localhost', 27017)
        admin = connection['admin']
        admin.authenticate('rrpdream', 'rrpdream9453')

        db = connection['KeywordFinder']
        search_results = db['project_terms'].find({'project_name': project_name}, {'keyword': True})
        results_list = list(search_results)
        keyword_list = results_list[0]['keyword']

        prediction_dict = dict()
        word_predictor = WordPredictor(self.config)
        for keyword in keyword_list:
            predict_list = word_predictor.find_similar_word(keyword, self.config['n_predict'])
            prediction_dict[keyword] = predict_list

        if output == 'mongo':
            time = str(datetime.now().date())
            document = {'time': time, 'prediction': prediction_dict}
            result = db['prediction'].insert_one(document)
            print(result)

        elif output == 'print':
            print(prediction_dict)

        elif output == 'json':
            with open(self.config['path'] + 'project_terms_prediction.json', 'w') as f:
                json.dump(prediction_dict, f)
