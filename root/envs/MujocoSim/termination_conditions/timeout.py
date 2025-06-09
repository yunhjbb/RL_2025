class Timeout():
    def __init__(self, config):
        self.max_step = config.get("max_steps", 10000)
        pass

    def get_termination(self, model, data, current_step, additional_data = None):
        if current_step >= self.max_step:
            done = True
            print("Timeout!", end = " ")
        else:
            done = False
        return done
