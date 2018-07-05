class Measure:
    def __init__(self):
        self.maxlike = -1*10**100
        self.score = 0

        # last performance the climber reach
        self.last_maxlike = None
        self.last_score = None

    def compare_maxlike(self, new_maxlike):
        if new_maxlike > self.maxlike:
            msg = 'better'

        else:
            msg = 'worse'

        return msg

    def compare_score(self, new_score):
        if new_score > self.score:
            msg = 'better'

        else:
            msg = 'worse'

        return msg
