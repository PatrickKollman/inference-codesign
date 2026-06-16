# Environment

Verified: 2026-06-16T04:39:51.071378+00:00
Raw artifact: `results/day1_env.json`

## GPU

```
Tue Jun 16 04:39:51 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.20             Driver Version: 580.126.20     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:02:00.0 Off |                  Off |
|  0%   30C    P8             11W /  450W |       1MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## CUDA Compiler

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

## Software Stack

| Package | Version |
|---------|---------|
| Python | `3.12.3` |
| PyTorch | `2.8.0+cu128` |
| CUDA (torch.version.cuda) | `12.8` |
| cuDNN | `91002` |
| ultralytics | `8.4.68` |

## Device Properties

| Property | Value |
|----------|-------|
| Device name | NVIDIA GeForce RTX 4090 |
| Compute capability | 8.9 |
| VRAM | 25.25 GB |

---
*All reported numbers in this project trace to this environment via `day1_env.json` timestamp.*
