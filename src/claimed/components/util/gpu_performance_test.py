#!/usr/bin/env python3

"""
PyTorch HPC Benchmark Script

Covers:
- CPU performance
- GPU single / multi GPU
- Distributed multi-node (DDP)
- DataLoader / IO throughput
- Synthetic data generation (lazy)
- Optional dataset materialization + cleanup
- Training + inference benchmarks

Usage examples:

Single GPU:
    python pytorch_hpc_benchmark.py --mode single_gpu

Multi GPU (single node):
    torchrun --nproc_per_node=4 pytorch_hpc_benchmark.py --mode ddp

Multi node:
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=... --master_port=... pytorch_hpc_benchmark.py --mode ddp

CPU only:
    python pytorch_hpc_benchmark.py --mode cpu
"""

import os
import time
import argparse
import shutil
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

# =====================
# Synthetic Dataset
# =====================

class SyntheticDataset(Dataset):
    def __init__(self, size, input_dim, num_classes, materialize_dir=None):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.materialize_dir = materialize_dir

        if materialize_dir:
            os.makedirs(materialize_dir, exist_ok=True)

    def __len__(self):
        return self.size

    def _generate(self, idx):
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y

    def __getitem__(self, idx):
        if self.materialize_dir:
            path = os.path.join(self.materialize_dir, f"{idx}.pt")
            if os.path.exists(path):
                return torch.load(path)
            else:
                sample = self._generate(idx)
                torch.save(sample, path)
                return sample
        else:
            return self._generate(idx)

# =====================
# Model
# =====================

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, depth=3):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =====================
# Utilities
# =====================

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def throughput(samples, elapsed):
    return samples / elapsed

# =====================
# Benchmarks
# =====================

def benchmark_dataloader(loader, device, steps):
    start = time.time()
    total = 0
    for i, (x, y) in enumerate(loader):
        if i >= steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        total += x.size(0)
    elapsed = time.time() - start
    return throughput(total, elapsed)


def benchmark_training(model, loader, device, steps):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    total = 0

    for i, (x, y) in enumerate(loader):
        if i >= steps:
            break
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total += x.size(0)

    elapsed = time.time() - start
    return throughput(total, elapsed)


def benchmark_inference(model, loader, device, steps):
    model.eval()
    total = 0
    start = time.time()

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= steps:
                break
            x = x.to(device)
            _ = model(x)
            total += x.size(0)

    elapsed = time.time() - start
    return throughput(total, elapsed)


def benchmark_cpu(matrix_size, iterations):
    a = torch.randn(matrix_size, matrix_size)
    b = torch.randn(matrix_size, matrix_size)

    start = time.time()
    for _ in range(iterations):
        _ = torch.mm(a, b)
    elapsed = time.time() - start

    flops = 2 * (matrix_size ** 3) * iterations
    return flops / elapsed / 1e9  # GFLOPS


def benchmark_gpu(matrix_size, iterations, device):
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(iterations):
        _ = torch.mm(a, b)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    flops = 2 * (matrix_size ** 3) * iterations
    return flops / elapsed / 1e9

# =====================
# Main
# =====================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["cpu", "single_gpu", "ddp"], required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset_size", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--input_dim", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--materialize_dir", type=str, default=None)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--matrix_size", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=50)

    args = parser.parse_args()

    if args.mode == "cpu":
        print("CPU GFLOPS:", benchmark_cpu(args.matrix_size, args.iterations))
        return

    if args.mode == "single_gpu":
        device = torch.device("cuda:0")
    elif args.mode == "ddp":
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")

    dataset = SyntheticDataset(
        args.dataset_size,
        args.input_dim,
        args.num_classes,
        materialize_dir=args.materialize_dir
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
    )

    model = SimpleMLP(
        args.input_dim,
        args.hidden_dim,
        args.num_classes,
        args.depth
    ).to(device)

    if args.mode == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    print("\n--- DataLoader throughput ---")
    dl_tp = benchmark_dataloader(loader, device, args.steps)
    print(f"Samples/sec: {dl_tp:.2f}")

    print("\n--- Training throughput ---")
    train_tp = benchmark_training(model, loader, device, args.steps)
    print(f"Samples/sec: {train_tp:.2f}")

    print("\n--- Inference throughput ---")
    infer_tp = benchmark_inference(model, loader, device, args.steps)
    print(f"Samples/sec: {infer_tp:.2f}")

    print("\n--- GPU compute ---")
    gpu_gflops = benchmark_gpu(args.matrix_size, args.iterations, device)
    print(f"GFLOPS: {gpu_gflops:.2f}")

    if args.cleanup and args.materialize_dir:
        shutil.rmtree(args.materialize_dir, ignore_errors=True)
        print("Materialized dataset removed.")

    if args.mode == "ddp":
        cleanup_ddp()


if __name__ == "__main__":
    main()
