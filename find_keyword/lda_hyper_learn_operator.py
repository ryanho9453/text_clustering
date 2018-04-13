import json

from term_document_matrix import Term_document_Matrix
from lda_colla_gibbs_model import Lda_Colla_Gibbs_model
from word_predictor import Word_Predictor
from evaluate_text_model import Evaluate_Model
from lda_hyper_learn_manager import Manager


class Operator():
    def __init__(self, config):
        self.config = config
        self.step = config['step']
        self.first_manager = Manager(first_run=True, config=config)
        self.regular_manager = Manager(first_run=False, config=config)

    def run(self):

        self._set_alpha(self.config['alpha'], self.config['beta'], self.first_manager)
        self._change_manager(self.first_manager, self.regular_manager)
        self._set_alpha(self.first_manager.recorder['opt_alpha'], self.first_manager.recorder['opt_beta'], self.regular_manager)


    def _set_alpha(self, alpha, beta, manager):
        print('--- set_alpha')
        print('alpha beta : '+str(alpha)+' '+str(beta))
        manager.recorder['route'].append((alpha, beta))

        adjust = manager.alpha_command(beta)

        alpha_up = alpha + self.step
        alpha_down = alpha - self.step

        if adjust == 'both':
            maxlike_up, score_up = self.__run_machine(alpha_up, beta, manager)
            maxlike_down, score_down = self.__run_machine(alpha_down, beta, manager)

            if score_up > score_down:
                new_alpha = alpha_up
                new_score = score_up
                adjust = 'up'

            else:
                new_alpha = alpha_down
                new_score = score_down
                adjust = 'down'

        elif adjust == 'up':
            new_maxlike, new_score = self.__run_machine(alpha_up, beta, manager)
            new_alpha = alpha_up

        elif adjust == 'down':
            new_maxlike, new_score = self.__run_machine(alpha_down, beta, manager)
            new_alpha = alpha_down

        next_step, next_param = manager.alpha_or_beta(beta=beta, new_score=new_score, move=adjust)

        if next_step == 'alpha':
            return self._set_alpha(new_alpha, beta, manager)

        elif next_step == 'beta' and next_param == 'old':
            self._set_beta(alpha, beta, manager)

        elif next_step == 'beta' and next_param == 'new':
            self._set_beta(new_alpha, beta, manager)


    def _set_beta(self, alpha, beta, manager):
        print('--- set_beta')
        print('alpha beta : ' + str(alpha) + ' ' + str(beta))
        manager.recorder['route'].append((alpha, beta))

        adjust = manager.beta_command(alpha)

        beta_up = beta + self.step
        beta_down = beta - self.step

        if adjust == 'both':
            maxlike_up, score_up = self.__run_machine(alpha, beta_up, manager)
            maxlike_down, score_down = self.__run_machine(alpha, beta_down, manager)

            if maxlike_up > maxlike_down:
                new_beta = beta_up
                new_maxlike = maxlike_up
                adjust = 'up'

            else:
                new_beta = beta_down
                new_maxlike = maxlike_down
                adjust = 'down'

        elif adjust == 'up':
            new_maxlike, new_score = self.__run_machine(alpha, beta_up, manager)
            new_beta = beta_up

        elif adjust == 'down':
            new_maxlike, new_score = self.__run_machine(alpha, beta_down, manager)
            new_beta = beta_down

        next_step, next_param = manager.alpha_or_beta(alpha=alpha, new_maxlike=new_maxlike, move=adjust)

        if next_step == 'beta' and next_param == 'new':
            return self._set_beta(alpha, new_beta, manager)

        elif next_step == 'alpha' and next_param == 'old':
            self._set_alpha(alpha, beta, manager)

        elif next_step == 'end':
            manager.recorder['opt_alpha'] = alpha
            manager.recorder['opt_beta'] = beta

            print('end tuning')

            manager.save_manager_record()



    def _change_manager(self, first, regular):
        print('change manager')

        regular.best_score = first.best_score
        regular.best_maxlike = first.best_maxlike
        regular.last_alpha_used_in_beta = first.last_alpha_used_in_beta
        regular.last_beta_used_in_alpha = first.last_beta_used_in_alpha
        regular.last_alpha_move = first.last_alpha_move
        regular.last_beta_move = first.last_beta_move
        regular.recorder = first.recorder


    def __run_machine(self, alpha, beta, manager):
        if str((alpha, beta)) not in manager.recorder['memory'].keys():
            machine = Machine(self.config)
            machine.run(alpha, beta)
            maxlike = machine.maxlike
            score = machine.score

            manager.recorder['memory'][str((alpha, beta))] = (maxlike, score)

            return maxlike, score

        else:
            print('machine already been here')
            maxlike = manager.recorder['memory'][str((alpha, beta))][0]
            score = manager.recorder['memory'][str((alpha, beta))][1]

            return maxlike, score



class Machine():
    def __init__(self, config):
        self.config = config

    def run(self, alpha, beta):
        with open(self.config['path'] + 'ground_truth.json', 'r', encoding='utf8') as f:
            ground_truth = json.load(f)

        matrix = Term_document_Matrix(self.config)
        td_matrix = matrix.load()

        model = Lda_Colla_Gibbs_model(self.config)
        model.build(td_matrix, alpha, beta, self.config['n_topics'])
        p_zw = model.get_p_zw()
        self.maxlike = model.get_maxlike()

        predictor = Word_Predictor(self.config, p_zw)

        evaluate = Evaluate_Model(self.config)
        self.score = evaluate.ground_truth(predictor, ground_truth)









