"""
climber is designed to explore the (alpha , beta) space
and try to reach the summit, the (alpha, beta) has the highest maximum likelihood and evaluation score

the function of climber is simple, "toward the summit" can be break into 3 parts

1. watch the mobile's guidance for next step, alpha or beta
2. walk
3. report the results to the mobile

repeat the process until the mobile says "goal"

"""

from .mobile import Mobile
from .gps import GPS
from .measure import Measure
from ..prepare_data.term_document_matrix import TermDocumentMatrix
from ..lda_collapsed_gibbs.model import LdaModel
from ..lda_collapsed_gibbs.word_predictor import WordPredictor
from ..lda_collapsed_gibbs.evaluate import Evaluate


class Climber:
    def __init__(self, config):
        self.config = config

        gps = GPS(config)
        measure = Measure()

        self.mobile = Mobile(gps, measure, config)

        self.step = config['step']

    def toward_summit(self):
        while self.mobile.goal is False:
            direction = self.mobile.guide()
            new_alpha, new_beta, new_maxlike, new_score = self._walk(direction)
            self.mobile.assess_results(new_alpha, new_beta, new_maxlike, new_score)

    def _walk(self, direction):
        if direction == 'alpha':
            new_alpha, new_beta, new_maxlike, new_score = self._set_alpha()

            return new_alpha, new_beta, new_maxlike, new_score

        elif direction == 'beta':
            new_alpha, new_beta, new_maxlike, new_score = self._set_beta()

            return new_alpha, new_beta, new_maxlike, new_score

    def _set_alpha(self):
        alpha_up = self.mobile.gps.alpha + self.step
        alpha_down = self.mobile.gps.alpha - self.step
        new_beta = None

        maxlike_up, score_up = self.__run_machine(alpha_up, self.mobile.gps.beta)
        maxlike_down, score_down = self.__run_machine(alpha_down, self.mobile.gps.beta)

        if score_up > score_down:
            new_alpha = alpha_up
            new_maxlike = maxlike_up
            new_score = score_up

        else:
            new_alpha = alpha_up
            new_maxlike = maxlike_down
            new_score = score_down

        return new_alpha, new_beta, new_maxlike, new_score

    def _set_beta(self):
        beta_up = self.mobile.gps.beta + self.step
        beta_down = self.mobile.gps.beta - self.step
        new_alpha = None

        maxlike_up, score_up = self.__run_machine(self.mobile.gps.alpha, beta_up)
        maxlike_down, score_down = self.__run_machine(self.mobile.gps.alpha, beta_down)

        if maxlike_up > maxlike_down:
            new_beta = beta_up
            new_maxlike = maxlike_up
            new_score = score_up

        else:
            new_beta = beta_down
            new_maxlike = maxlike_down
            new_score = score_down

        return new_alpha, new_beta, new_maxlike, new_score

    def __run_machine(self, alpha, beta):
        if str((alpha, beta)) not in self.mobile.memory.keys():
            machine = Machine(self.config)
            machine.run(alpha, beta)
            score = machine.score
            maxlike = machine.maxlike

            self.mobile.memory[str((alpha, beta))] = (maxlike, score)

            return maxlike, score

        else:
            print('machine already been here')
            maxlike = self.mobile.memory[str((alpha, beta))][0]
            score = self.mobile.memory[str((alpha, beta))][1]

            return maxlike, score


class Machine:
    def __init__(self, config):
        self.config = config
        self.maxlike = None
        self.score = None

    def run(self, alpha, beta):
        matrix = TermDocumentMatrix(self.config)
        td_matrix = matrix.load()

        model = LdaModel(self.config)
        model.build(td_matrix, alpha, beta, self.config['n_topics'])
        p_zw = model.get_p_zw()
        self.maxlike = model.get_maxlike()

        predictor = WordPredictor(self.config, p_zw)

        evaluate = Evaluate(self.config)
        self.score = evaluate.ground_truth(predictor)
