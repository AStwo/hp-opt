from src.optimizer.opt.BaseOptimizer import BaseOptimizer


class RandomOptimizer(BaseOptimizer):
    def __init__(self, search_space, seed=None):
        super().__init__(search_space, seed)

    def optimize(self, eval_function, iterations, metric_target=None, early_stop=None, objective="min"):
        assert objective in ("min", "max")

        # Prepare variables
        early_stop_counter = 0
        i = 0
        sign = 1 if objective == "max" else -1
        if metric_target is not None:
            metric_target *= sign

        while True:
            proposed_solution = self.get_random_point(self.search_space)
            proposed_target = sign * eval_function(**proposed_solution)

            self.hist_params.append(proposed_solution)
            self.hist_target.append(sign * proposed_target)

            # Update best target
            try:
                if proposed_target >= self.best_target:
                    self.best_target = proposed_target
                    self.best_solution = proposed_solution
                    early_stop_counter = 0
            except TypeError:
                # self.best_target = proposed_target
                # self.best_solution = proposed_solution
                self.best_target = proposed_target
                self.best_solution = proposed_solution

            if self.optimization_stop_conditions(i, iterations, early_stop, early_stop_counter,
                                                 self.hist_target, metric_target):
                self.best_target *= sign
                break
            else:
                early_stop_counter += 1
                i += 1

    @staticmethod
    def get_random_point(search_space):
        return {param.name: param.get_value() for param in search_space.values()}
