"""
mobile got 2 application, gps and measure

gps is about the climber's location in (alpha, beta)

measure is about the maximum likelihood and evaluation score on (alpha, beta)

mobile use both app to support the climber

"""

import json


class Mobile:

    def __init__(self, gps, measure, config):
        self.first_run = True
        self.goal = False

        self.config = config

        # instances (app of the mobile)
        self.gps = gps
        self.measure = measure

        # memorize the performance of every point
        self.memory = dict()

        # record the route
        self.route = list()

        # record the results of every alpha beta step
        # tracking = [....., 'better_alpha', 'better_beta', ...]
        self.tracking = list()

    def guide(self):
        """
        the "guide" function tell the climber the next step to go, according to "last_step" and "last_performance"

        beyond that, there are 2 phase in the climbing, "first_run" and "regular_run"

        in first run,
        the climber step alpha as far as it can , and then do the same thing with beta
        once it reach the best beta , it turns to "regular_run"

        in regular run,
        one cycle composed of one alpha step and several beta step to find the best beta for the alpha


        """

        # first run
        if self.first_run:
            print('first run')
            if self.gps.last_step is None:
                direction = 'alpha'

            elif self.gps.last_step == 'alpha':
                if self.measure.last_score == 'better':
                    direction = 'alpha'

                elif self.measure.last_score == 'worse':
                    direction = 'beta'

            elif self.gps.last_step == 'beta':
                if self.measure.last_maxlike == 'better':
                    direction = 'beta'

                elif self.measure.last_maxlike == 'worse':
                    self.first_run = False
                    print('end of first run')

                    return self.guide()

        # regular run
        else:
            print('regular run')
            if self.gps.last_step == 'alpha':
                direction = 'beta'

            elif self.gps.last_step == 'beta':
                if self.measure.last_maxlike == 'better':
                    direction = 'beta'

                elif self.measure.last_maxlike == 'worse':
                    direction = 'alpha'

        print('step : '+str(direction))

        return direction

    def assess_results(self, new_alpha, new_beta, new_maxlike, new_score):
        """
        update the info in gps, measure, according to how the step does

        if 'better' , update the better alpha/beta to gps and new (maxlike, score) to measure

        on the other hand, keep records of the climbing
        "last_step", "last_score", "last_maxlike"  --- for the last step

        "tracking" -- track each step's result , use this to decide the ending

        "route" -- track the climber's walking path

        """

        if new_alpha:
            msg = self.measure.compare_score(new_score)
            if msg == 'better':
                self.gps.alpha = new_alpha
                self.measure.maxlike = new_maxlike
                self.measure.score = new_score

                self.gps.last_step = 'alpha'
                self.measure.last_score = 'better'

                self.tracking.append('better_alpha')
                self.route.append((self.gps.alpha, self.gps.beta))

            elif msg == 'worse':
                self.gps.last_step = 'alpha'
                self.measure.last_score = 'worse'

                self.tracking.append('worse_alpha')
                self.route.append((self.gps.alpha, self.gps.beta))

                # check if end
                self._check_if_goal()

        elif new_beta:
            msg = self.measure.compare_maxlike(new_maxlike)
            if msg == 'better':
                self.gps.beta = new_beta
                self.measure.maxlike = new_maxlike
                self.measure.score = new_score

                self.gps.last_step = 'beta'
                self.measure.last_maxlike = 'better'

                self.tracking.append('better_beta')
                self.route.append((self.gps.alpha, self.gps.beta))

            elif msg == 'worse':
                self.gps.last_step = 'beta'
                self.measure.last_maxlike = 'worse'

                self.tracking.append('worse_beta')
                self.route.append((self.gps.alpha, self.gps.beta))

        print('results : '+str(msg))
        print('location : '+str((self.gps.alpha, self.gps.beta)))

    def _check_if_goal(self):

        """
        given the best beta for a given alpha has found, and we can't find a better alpha further
        we will say the climbing comes to an end

        which means the end must fulfill the belows:

        1. it ends on "regular run" phase

        2. it ends after a worse alpha step -- means couldn't find a better alpha

        3. just before the worse alpha, there is a worse beta --- means it has found the best beta for alpha

        "self.tracking" record the results of each step, something like
        [......, worse_maxlike, worse_score]
                      ||             ||
                      ||           no alpha(n)
                best beta for alpha(n-1)

        """

        if self.tracking[-2:] == ['worse_beta', 'worse_alpha']:
            self.goal = True
            print('ending -- reach the summit')

            # save the record and results
            journal = dict()
            journal['memory'] = self.memory
            journal['route'] = self.route
            journal['opt_alpha'] = self.gps.alpha
            journal['opt_beta'] = self.gps.beta
            journal['maxlike'] = self.measure.maxlike
            journal['score'] = self.measure.score

            with open(self.config['path'] + 'journal' + self.config['model_ver'] + '.json', 'w', encoding='utf8') as f:
                json.dump(journal, f)

            print('end of climbing')
            print('opt_alpha : '+str(self.gps.alpha))
            print('opt_beta : ' + str(self.gps.beta))
            print('maxlike : '+str(self.measure.maxlike))
            print('score : '+str(self.measure.score))
