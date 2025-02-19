# Plotting

Results over several benchmarks can be visualized in two ways.

## After HPO

After running HPO, you make download the json file stored as an artifact at the top level mlflow run using `plotting/plot_results_mlflow.ipynb`.

## After repeated runs

After doing repeated runs, you can use `plotting/plot_results_repeated_runs.ipynb` using the file produced by `repeat_best_experiment`.
