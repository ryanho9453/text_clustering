class GPS:
    def __init__(self, config):
        self.alpha = config['alpha']
        self.beta = config['beta']

        # last action
        self.last_step = None
