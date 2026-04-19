```yaml
prompts_used: prompts/clarity.txt
my_reads: 
    - ltrc/infra_reccs.md
    - ltrc/state_of_infra.md
    - meeting_notes/meeting0.md
    - knowledge/primer.md
```

# 0env - setting up the test bed

the single question this doc answers: how do i get a working GPU environment on the cluster so i can run, train, and serve models?

---

## questions to ask before touching anything

for supervisor:

1. what exactly am i here to do - what is the problem this team is trying to solve?
2. what does a successful end to my internship look like?
3. which machine do i log into, and how - is there a VPN, an SSH key, a specific hostname?
4. once i'm in, what am i allowed to install and where? is anything off-limits?
5. how much disk space do i actually have, and where should my code, models, and data live?
6. is there already a working environment i should build on, or am i starting from scratch?
7. what model am i supposed to run first - what is the actual task?
8. how long can a job run before the system kills it? what happens to my work when it does?
9. if something breaks on the cluster itself - a GPU is down, a mount is missing - who do i tell?
10. is there a shared place where the team's models and datasets are stored, so i don't re-download what already exists?

for fellow interns:

1. what are you working on, and how does my work connect to yours?
2. what broke when you were setting up, and how did you fix it?
3. is there a shared environment, Docker image, or conda prefix i should use?
4. where are your cached models sitting - can i point to the same cache?
5. is there a repo i should clone before i write any code?

---

## storage constraints (read before installing anything)

> why these limits exist and what's being done about them: primer.md § "The Problems with Our Cluster Right Now"

| location | size | persists? | use for |
|---|---|---|---|
| `~/` (home) | 30GB | yes | dotfiles, bashrc only |
| `/share1/$USER/` | 100GB | yes | env, model cache, checkpoints |
| `/tmp/` | 1TB | no - 7 day delete | scratch only |

everything goes in share1. home fills up fast. tmp disappears.

---

## step 1: assess the node

> what Ada and Turing nodes actually have, and why GPU model matters: primer.md § "The Two Clusters - Ada and Turing"

```bash
whoami && hostname
nvidia-smi                    # GPU model + VRAM + driver
nvcc --version                # CUDA version - must match torch build
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
df -h                         # check free space on home and share1
ls /share1/                   # check if teammates have a shared cache already
```

the 1080 nodes only support older CUDA. if you install a torch built for CUDA 12.x on a 1080, it silently falls back to CPU. match nvcc output to the torch install below.

---

## step 2: set cache dirs

```bash
echo 'export HF_HOME=/share1/$USER/.cache/hf' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/share1/$USER/.cache/hf' >> ~/.bashrc
echo 'export PIP_CACHE_DIR=/share1/$USER/.cache/pip' >> ~/.bashrc
echo 'export CONDA_PKGS_DIRS=/share1/$USER/.cache/conda' >> ~/.bashrc
source ~/.bashrc
```

---

## step 3: create env in share1

> why model size determines how much space you need: primer.md § "What '11GB VRAM' means in practice" (precision table)

```bash
conda create --prefix /share1/$USER/envs/main python=3.10 -y
conda activate /share1/$USER/envs/main

# match CUDA version from nvcc --version - example for CUDA 11.8:
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets peft
```

space check after:

```bash
du -sh /share1/$USER/envs/main /share1/$USER/.cache/
df -h /share1
```

env + cache should be under 40GB total or you won't have room for models.

---

## step 4: verify GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```

---

## step 5: dummy inference (proof of life)

> what happens when you call this pipeline - prefill then decode: primer.md § "The Two Phases of LLM Inference"

check if a shared cache exists first - don't re-download what's already there:

```bash
ls /share1/  # or ask teammates where their HF_HOME points
```

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="facebook/opt-125m", device=0)
print(pipe("Hello, I am", max_new_tokens=20))
```

model must download to share1, not home.

---

## step 6: verify all 4 GPUs

> why 4 GPUs on one node is the safe limit right now, and what InfiniBand has to do with it: primer.md § "Layer 5: Distributed Orchestration" (NVLink vs InfiniBand diagram)

```python
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory/1e9:.1f}GB")
```

cross-node training is currently bottlenecked - InfiniBand is underperforming. single-node 4-GPU is the ceiling until that's fixed.

---

## step 7: multi-GPU smoke test

> what DDP actually does when all 4 ranks start up: primer.md § "Training Terms" (DDP, NCCL, gradient)

```bash
torchrun --nproc_per_node=4 -m torch.distributed.launch --use_env test_ddp.py
```

minimal test_ddp.py:

```python
import torch, torch.distributed as dist
dist.init_process_group("nccl")
rank = dist.get_rank()
print(f"rank {rank} / {dist.get_world_size()} - GPU {torch.cuda.current_device()}")
dist.destroy_process_group()
```

---

## first week checklist

- [ ] env working, CUDA version matched, confirmed with teammates on torch version
- [ ] HF cache lands in share1 - verified with `ls /share1/$USER/.cache/hf/`
- [ ] opt-125m inference runs end to end
- [ ] LoRA finetune on toy dataset runs (proves training pipeline works) - what LoRA is: primer.md § "Layer 7: Model Optimization" (LoRA/PEFT row)
- [ ] checkpoint saves mid-run, reloads, continues - non-negotiable given 4-day wall-time (primer.md § "The Compute Problem in Detail")
- [ ] all 4 GPUs verified, torchrun multi-GPU test passes
- [ ] everything that broke is written down

---

## next: containerization

> where containerization fits in the full work plan: primer.md § "The Work Plan - What You Will Actually Do" (Step 2)

once bare-metal env is stable, containerize it. this is required for portability to Turing and cloud.

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
RUN pip install transformers==4.40.0 accelerate datasets peft
ENV HF_HOME=/share1/$USER/.cache/hf
WORKDIR /workspace
```
