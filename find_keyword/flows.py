from .prepare_data.term_document_matrix import TermDocumentMatrix
from .lda_collapsed_gibbs.model import LdaModel
from .lda_collapsed_gibbs.word_predictor import WordPredictor
from .lda_collapsed_gibbs.evaluate import Evaluate
from .parameter_learn.climber import Climber


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

        predictor = WordPredictor(self.config, p_zw)

        evaluate = Evaluate(self.config)
        evaluate.ground_truth(predictor)

    def build_and_evaluate(self):
        matrix = TermDocumentMatrix(self.config)
        td_matrix = matrix.load()

        model = LdaModel(self.config)
        model.build(td_matrix, self.config['alpha'], self.config['beta'], self.config['n_topics'], save_model=True)
        p_zw = model.get_p_zw()

        predictor = WordPredictor(self.config, p_zw)

        evaluate = Evaluate(self.config)
        evaluate.ground_truth(predictor)

    def tune_alpha_beta(self):
        climber = Climber(self.config)
        climber.toward_summit()
