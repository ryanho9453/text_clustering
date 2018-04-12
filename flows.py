import json


from term_document_matrix import Term_document_Matrix
from lda_colla_gibbs_model import Lda_Colla_Gibbs_model
from word_predictor import Word_Predictor
from evaluate_text_model import Evaluate_Model
from lda_hyper_learn_operator import Operator



class Flows():
    def __init__(self, config):
        self.config = config

    def prepare_data(self):
        matrix = Term_document_Matrix(self.config)
        matrix.create(save_matrix=True)


    def prepare_build_and_evaluate(self):
        with open(self.config['path'] + 'ground_truth.json', 'r', encoding='utf8') as f:
            ground_truth = json.load(f)

        matrix = Term_document_Matrix(self.config)
        td_matrix = matrix.create(save_matrix=True)

        model = Lda_Colla_Gibbs_model(self.config)
        model.build(td_matrix, self.config['alpha'], self.config['beta'], self.config['n_topics'], save_model=True)
        p_zw = model.get_p_zw()

        predictor = Word_Predictor(self.config, p_zw)

        evaluate = Evaluate_Model(self.config)
        evaluate.ground_truth(predictor, ground_truth)


    def build_and_evaluate(self):
        with open(self.config['path'] + 'ground_truth.json', 'r', encoding='utf8') as f:
            ground_truth = json.load(f)

        matrix = Term_document_Matrix(self.config)
        td_matrix = matrix.load()

        model = Lda_Colla_Gibbs_model(self.config)
        model.build(td_matrix, self.config['alpha'], self.config['beta'], self.config['n_topics'], save_model=True)
        p_zw = model.get_p_zw()

        predictor = Word_Predictor(self.config, p_zw)

        evaluate = Evaluate_Model(self.config)
        evaluate.ground_truth(predictor, ground_truth)

    def tune_alpha_beta(self):
        operator = Operator(self.config)
        operator.run()













