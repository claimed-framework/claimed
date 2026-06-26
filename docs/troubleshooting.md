# Troubleshooting

## Corrupted MLFlow runs

Sometimes, especially in experiments with many runs, MLFlow runs seem to get corrupted.

This causes the UI to not be able to display any runs for the containing experiment.

The solution is to identify the corrupted run and delete it.

To more easily do this, you may use the script `mlflow_corrupted.py`.

```sh
python mlflow_corrupted.py
```

will display the usage instructions.

## status code 500: No available agent

The error `RuntimeError: Request failed with status code 500: No available agent to submit job, please try again later.. ` on job submission may happen when two users are sharing the same machine for their head node. This may also happen if you are running two ray clusters simultaneously.

This is technically not supported by ray, however, in this case, you may solve this by adding `--dashboard-agent-listen-port <unique port>` to your `ray start` command, with a different port for each ray head.

