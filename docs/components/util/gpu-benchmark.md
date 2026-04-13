# gpu_performance_test

PyTorch HPC benchmark component covering CPU, single-GPU, and multi-node distributed (DDP) workloads.

## CLI

```bash
claimed run claimed.components.util.gpu_performance_test --help
```

```bash
# CPU matrix-multiply benchmark
claimed run claimed.components.util.gpu_performance_test \
    --mode cpu \
    --matrix-size 4096 \
    --iterations 100

# Single GPU full benchmark
claimed run claimed.components.util.gpu_performance_test \
    --mode single_gpu \
    --steps 50

# Multi-node DDP (via torchrun)
torchrun --nnodes=2 --nproc_per_node=4 \
    -m claimed.components.util.gpu_performance_test \
    --mode ddp
```

## Benchmark Phases

| Phase | Metric | Description |
|---|---|---|
| DataLoader throughput | samples/sec | Measures IO / preprocessing pipeline speed |
| Training throughput | samples/sec | Forward + backward + optimiser step |
| Inference throughput | samples/sec | Forward pass only, `torch.no_grad()` |
| GPU compute | GFLOPS | Dense matrix-multiply (`torch.mm`) |
| CPU compute | GFLOPS | Same on CPU tensors |

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mode` | str | `single_gpu` | `cpu` \| `single_gpu` \| `ddp` |
| `batch_size` | int | 256 | DataLoader batch size |
| `num_workers` | int | 4 | DataLoader worker processes |
| `dataset_size` | int | 100 000 | Total synthetic samples |
| `steps` | int | 100 | Batches per benchmark phase |
| `input_dim` | int | 1 024 | MLP input feature dimension |
| `hidden_dim` | int | 2 048 | MLP hidden layer width |
| `num_classes` | int | 10 | Output classes |
| `depth` | int | 3 | Number of hidden layers |
| `materialize_dir` | str | `None` | Cache synthetic data on disk |
| `cleanup` | bool | `False` | Delete `materialize_dir` after benchmark |
| `matrix_size` | int | 2 048 | Square matrix edge for compute test |
| `iterations` | int | 50 | Matrix-multiply iterations |

## Python API

::: claimed.components.util.gpu_performance_test
    options:
      members:
        - run
        - benchmark_cpu
        - benchmark_gpu
        - benchmark_training
        - benchmark_inference
        - benchmark_dataloader
