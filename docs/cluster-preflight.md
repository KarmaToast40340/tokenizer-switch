# Read-only cluster preflight — 6 November 2026

No project GPU job was submitted. Later checks were read-only: no further
installation, coursework deletion or job changes.

`/opt/conda/miniconda3/envs/DL_HW` imports torch, transformers, tokenizers, YAML
and NumPy successfully. Python 3.12.7; torch 2.6.0+cu124; transformers 4.49.0;
tokenizers 0.21.0. CPU/library compatibility is present. CUDA is unavailable on
the login node, as expected without an allocation; this is not a GPU test.

Compiled architectures: sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90.
The scheduler confirms RTX 5050 nodes, matching the planned hardware. RTX 5050
is compute capability 12.0; this build lacks sm_120. Official Blackwell support
arrived with PyTorch 2.7/CUDA 12.8. Thus this environment is not a verified or
supported runtime for the planned RTX 5050 experiments. We did not modify it
or silently switch experiments to older GPUs.

Sources: [NVIDIA capabilities](https://developer.nvidia.com/cuda/gpus),
[PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/).

Home is near its 10 GB quota. Private diagnostics remain outside Git. Chinese
transfer files total 743.1 MiB. Add about 0.5 GiB GPT-2 weights, about 0.5 GiB
per retained checkpoint, plus logs/headroom. About **2 GiB additional persistent project space** covers assets/cache and one checkpoint; retaining all three A/B/E checkpoints needs **3 GiB** with an approved shared CUDA 12.8 environment.
A private environment would need several more GiB; plan approximately
**8–10 GiB extra total**, an estimate rather than a measured completed install.

Exact question for Djam or the administrator:

> My course home is near its 10 GB limit. Where may I store about 3 GiB of
> tokenizer-project data, GPT-2 cache and checkpoints on persistent storage
> accessible to compute nodes, and which existing environment supports RTX
> 5050 (sm_120), PyTorch with CUDA 12.8, transformers and tokenizers? If there
> is no shared environment, can I obtain roughly 8–10 GiB additional quota for an
> isolated environment plus assets without deleting other coursework?

Before Chinese training: approved storage/runtime; copy and checksum assets;
cache GPT-2; run a small Slurm GPU compatibility/memory test; recheck queue and
duplicate run directories; submit one assigned 50M job at a time. Chinese 200M
B/E still need expanded corpora. The trainer does not support an unchanged
50M-to-200M continuation with optimizer/RNG state.
