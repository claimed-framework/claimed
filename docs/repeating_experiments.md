# Repeating best experiments

Following GeoBench, once HPO is done, you may wish to repeat the best parameters found for each task while varying the seeds.

You can do this with:

``` sh
repeat_best_experiment <parent_run_id> <absolute_path_to_store_results> --config <path_to_config>
```

## :::benchmark.repeat_best_experiment.rerun_best_from_backbone
