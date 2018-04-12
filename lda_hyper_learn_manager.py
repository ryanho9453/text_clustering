import json


class Manager():
    def __init__(self, config, first_run):
        self.config = config

        self.first_run = first_run

        self.last_alpha_used_in_beta = None
        self.last_beta_used_in_alpha = None
        self.last_alpha_move = None
        self.last_beta_move = None

        self.best_score = 0
        self.best_maxlike = -1*10**100

        self.recorder = {}
        self.recorder['route'] = []
        self.recorder['memory'] = {}
        self.recorder['opt_alpha'] = None
        self.recorder['opt_beta'] = None

    def alpha_command(self, beta):
        if self.first_run is True:
            print('---- start first run')
            if self.last_alpha_move is None:
                command = 'both'

            else:
                command = self.last_alpha_move

        else:
            print('---- start regular run')
            beta_report = self._compare_beta(beta)

            if beta_report == 'pro':
                command = 'both'

            elif beta_report == 'no_pro':
                command = self.last_alpha_move

        return command


    def beta_command(self, alpha):
        if self.first_run is True:
            if self.last_beta_move is None:
                command = 'both'

            else:
                command = self.last_beta_move

        else:
            alpha_report = self._compare_alpha(alpha)

            if alpha_report == 'pro':
                command = 'both'

            elif alpha_report == 'no_pro':
                command = self.last_beta_move

        return command

    def alpha_or_beta(self, alpha=None, beta=None, new_score=None, new_maxlike=None, move=None):
        if self.first_run is True:
            # set alpha
            if beta is not None and new_score is not None:
                score_report = self._compare_score(new_score)
                self.last_beta_used_in_alpha = beta

                if score_report == 'worse':
                    next_step = 'beta'
                    next_param = 'old'

                elif score_report == 'better':
                    next_step = 'alpha'
                    next_param = 'new'
                    self.last_alpha_move = move
                    self.best_score = new_score

                print(score_report)

            # set beta
            elif alpha is not None and new_maxlike is not None:
                maxlike_report = self._compare_maxlike(new_maxlike)
                self.last_alpha_used_in_beta = beta

                if maxlike_report == 'better':
                    next_step = 'beta'
                    next_param = 'new'
                    self.last_beta_move = move
                    self.best_maxlike = new_maxlike

                elif maxlike_report == 'worse':
                    next_step = 'end'
                    next_param = 'old'

                print(maxlike_report)

        else:
            # set alpha
            if beta is not None and new_score is not None:
                score_report = self._compare_score(new_score)
                self.last_beta_used_in_alpha = beta

                if score_report == 'worse':
                    next_step = 'beta'
                    next_param = 'old'

                elif score_report == 'better':
                    next_step = 'beta'
                    next_param = 'new'
                    self.last_alpha_move = move
                    self.best_score = new_score

                print(score_report)

            # set beta
            elif alpha is not None and new_maxlike is not None:
                alpha_report = self._compare_alpha(alpha)
                maxlike_report = self._compare_maxlike(new_maxlike)

                if maxlike_report == 'better':
                    next_step = 'beta'
                    next_param = 'new'
                    self.last_beta_move = move
                    self.best_maxlike = new_maxlike

                # alpha get better perform last time, give it another chance
                elif maxlike_report == 'worse' and alpha_report == 'pro':
                    next_step = 'alpha'
                    next_param = 'old'
                    self.last_alpha_used_in_beta = alpha

                # alpha didn't get good use of the chance, still no_pro , end of tuning
                elif maxlike_report == 'worse' and alpha_report == 'no_pro':
                    next_step = 'end'
                    next_param = 'old'

                print(maxlike_report)
                print(alpha_report)

        return next_step, next_param

    def save_manager_record(self):
        manager_record = {}
        manager_record['route'] = self.recorder['route']
        manager_record['memory'] = self.recorder['memory']
        manager_record['opt_alpha'] = self.recorder['opt_alpha']
        manager_record['opt_beta'] = self.recorder['opt_beta']
        manager_record['best_score'] = self.best_score
        manager_record['best_maxlike'] = self.best_maxlike

        print('opt_alpha ' + str(self.recorder['opt_alpha']))
        print('opt_beta ' + str(self.recorder['opt_beta']))
        print('best_score ' + str(self.best_score))
        print('best_maxlike ' + str(self.best_maxlike))

        with open(self.config['path'] + 'manager_record' + self.config['model_ver'] + '.json', 'w', encoding='utf8') as f:
            json.dump(manager_record, f)


    def _compare_alpha(self, alpha):
        if alpha != self.last_alpha_used_in_beta:
            alpha_report = 'pro'

        else:
            alpha_report = 'no_pro'

        print(alpha_report)
        return alpha_report

    def _compare_beta(self, beta):
        if beta != self.last_beta_used_in_alpha:
            beta_report = 'pro'

        else:
            beta_report = 'no_pro'

        print(beta_report)
        return beta_report

    def _compare_score(self, new_score):
        if new_score > self.best_score:
            score_report = 'better'

        else:
            score_report = 'worse'

        print(score_report)
        return score_report

    def _compare_maxlike(self, new_maxlike):
        if new_maxlike > self.best_maxlike:
            maxlike_report = 'better'

        else:
            maxlike_report = 'worse'

        print(maxlike_report)
        return maxlike_report


