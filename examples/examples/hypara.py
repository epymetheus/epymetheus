import sys

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn


def print_as_comment(obj):
    print("\n".join(f"# {line}" for line in str(obj).splitlines()))


if __name__ == "__main__":
    sys.path.append("../..")
    seaborn.set_style("whitegrid")

    # ---

    import pandas as ep

    import epymetheus as ep
    from epymetheus.benchmarks import dumb_strategy
    from epymetheus.datasets import fetch_usstocks

    universe = fetch_usstocks()

    def objective(trial):
        profit_take = trial.suggest_int("profit_take", 10, 100)
        stop_loss = trial.suggest_int("stop_loss", -100, -10)
        my_strategy = ep.create_strategy(
            dumb_strategy,
            profit_take=profit_take,
            stop_loss=stop_loss,
        )
        my_strategy.run(universe, verbose=False)

        return my_strategy.score("final_wealth")

    # study = optuna.create_study(direction="maximize")
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42)
    )
    study.optimize(objective, n_trials=100)

    print(">>> study.best_params")
    print_as_comment(study.best_params)
