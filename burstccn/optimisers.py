class CombinedOptimiser:
    def __init__(self, optimiser_list, lr_scheduler_list=None):
        self.optimisers = [opt for opt in optimiser_list if opt is not None]
        if lr_scheduler_list:
            self.lr_scheduler_list = lr_scheduler_list

    def step(self):
        for optimiser in self.optimisers:
            optimiser.step()

    def zero_grad(self):
        for optimiser in self.optimisers:
            optimiser.zero_grad()

    def step_schedule(self):
        for lr_scheduler in self.lr_scheduler_list:
            if lr_scheduler is not None:
                lr_scheduler.step()